import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import yaml
import logging
import pandas as pd
import numpy as np
import torch
import warnings
import wandb
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Import our modules
from utils.data_loader import load_data
from utils.preprocessing import compute_log_returns, normalize_series, denoise_signal
from analysis.takens_embedding import (
    compute_delay_mutual_information, 
    false_nearest_neighbors, 
    takens_embedding,
    estimate_tau,
    estimate_embedding_dim
)
from analysis.chaos_metrics import get_chaos_metrics
from analysis.lyapunov import compute_largest_lyapunov_exponent
from regime_detection.hmm_model import train_hmm, predict_regimes
from regime_detection.changepoint_detection import detect_changepoints_pelt
from dynamics.sindy_model import fit_sparse_dynamics, discover_equations
from dynamics.neural_ode_model import ODEFunc, NeuralODE
from training.train_dynamics import train_neural_ode
from visualization.attractor_plots import plot_2d_attractor, plot_3d_attractor
from visualization.recurrence_plots import compute_and_plot_recurrence
from visualization.phase_diagrams import plot_regime_segmentation, plot_changepoints

def create_synthetic_data(file_path: str):
    """Creates a synthetic chaotic time series (Lorenz x-component) for testing"""
    logger.info("Generating synthetic Lorenz data for testing...")
    from scipy.integrate import odeint
    def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
    
    t = np.arange(0.0, 40.0, 0.01)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(lorenz, state0, t)
    
    # Add some noise to make it "market-like"
    np.random.seed(42)
    price = np.exp(normalize_series(states[:, 0]) * 0.5 + 5) + np.random.normal(0, 0.5, len(t))
    
    df = pd.DataFrame({'time': t, 'price': price})
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path

def simulate_ou(alpha=5, mu=48.22, sigma=5, x0=62.24, n=5000):
    """Simulates an Ornstein-Uhlenbeck process for fast synthetic testing."""
    logger.info(f"Simulating OU Process (alpha={alpha}, mu={mu}, sigma={sigma}, x0={x0})")
    dt = 0.01
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] + alpha*(mu - x[i-1])*dt + sigma*np.sqrt(dt)*np.random.randn()
    
    t = np.arange(0, n*dt, dt)
    return pd.DataFrame({'time': t, 'price': x})

def get_args():
    parser = argparse.ArgumentParser(description="Nonlinear Market Dynamics Pipeline")
    parser.add_argument('--test_mode', type=str, choices=['synthetic', 'real'], default=None, 
                        help="Run in test mode with fast data (synthetic OU or short real dataset).")
    parser.add_argument('--gamma', type=float, default=None, help="Override risk aversion gamma.")
    parser.add_argument('--convenience_yield', type=float, default=None, help="Override convenience yield.")
    parser.add_argument('--tau', type=int, default=5, help="Override Takens delay tau.")
    parser.add_argument('--embedding_dim', type=int, default=4, help="Override Takens embedding dimension m.")
    parser.add_argument('--use_returns', action='store_true', help="Use optional returns transformation.")
    parser.add_argument('--sweep', action='store_true', help="Run sweep experiment over tau and m.")
    parser.add_argument('--experiment', action='store_true', help="Run expanded experiment over tau, m, and epsilon.")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs for Neural ODE.")
    parser.add_argument('--epsilon_percentile', type=float, default=10.0, help="Threshold percentile for recurrence plot.")
    parser.add_argument('--sindy_threshold', type=float, default=0.02, help="SINDy optimizer threshold.")
    parser.add_argument('--sindy_degree', type=int, default=3, help="SINDy polynomial degree.")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Load config
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    use_wandb = config['wandb']['mode'] == 'online'
    if use_wandb:
        wandb.init(project=config['wandb']['project'], config=config)
        
    # Data Loading / Processing based on arguments
    if args.test_mode == 'synthetic':
        df = simulate_ou()
        logger.info("Using synthetic OU data for testing.")
    elif args.test_mode == 'real':
        logger.info("Downloading real Brent daily data for testing (first 5000 points)...")
        df = pd.read_csv("https://raw.githubusercontent.com/datasets/oil-prices/master/data/brent-daily.csv")
        df = df.rename(columns={'Price': 'price'})
        time = pd.to_datetime(df["Date"])
        time = (time - time.iloc[0]).dt.days.astype(float)
        assert isinstance(time.iloc[0], float) or isinstance(time.iloc[0], (int, float))
        df["time"] = time
        df = df.dropna()
        df = df.head(5000).reset_index(drop=True)
    else:
        # Generate data if not exists
        raw_path = config['data']['raw_path']
        if not Path(raw_path).exists():
            create_synthetic_data(raw_path)

        # 2. Data Loading
        df = load_data(raw_path, time_col=config['data']['time_col'])
    
    # 3. Preprocessing
    logger.info("Applying data preprocessing: log returns and normalization only.")
    price = df[config['data'].get('target_col', 'price')]
    
    returns = np.diff(np.log(price))
    returns = pd.Series(returns)
    returns = (returns - returns.mean()) / returns.std()
    returns = np.clip(returns, -5, 5)
    
    # Align dataframe
    df = df.iloc[-len(returns):].copy()
    normalized_signal = returns.values
    
    # 4. Phase Space Configurations
    import itertools
    if getattr(args, 'experiment', False):
        param_grid = list(itertools.product([3, 5, 10], [3, 4, 5], [5.0, 10.0, 20.0]))
    elif getattr(args, 'sweep', False):
        param_grid = list(itertools.product([3, 5, 10], [3, 4, 5], [args.epsilon_percentile]))
    else:
        param_grid = [(args.tau, args.embedding_dim, args.epsilon_percentile)]
        
    for tau, m, ep in param_grid:
        logger.info(f"--- Running pipeline for tau={tau}, m={m}, epsilon_p={ep} ---")
        
        output_dir = f"outputs_tau{tau}_m{m}_ep{ep}" if (getattr(args, 'sweep', False) or getattr(args, 'experiment', False)) else ("/kaggle/working" if Path("/kaggle/working").exists() else "outputs")
        Path(output_dir).mkdir(exist_ok=True)
        log_file = f"outputs_tau{tau}_m{m}_ep{ep}.txt" if (getattr(args, 'sweep', False) or getattr(args, 'experiment', False)) else None

        def append_log(text):
            if log_file:
                with open(log_file, "a") as lf:
                    lf.write(text + "\n")
        
        embedded = takens_embedding(normalized_signal, tau=tau, m=m)
        logger.info(f"Phase space reconstructed. Shape: {embedded.shape}")
        
        # 5. Chaos Metrics
        logger.info("Computing chaos metrics...")
        metrics = get_chaos_metrics(normalized_signal, m=m, from_pyunicorn=False)
        lle = compute_largest_lyapunov_exponent(normalized_signal, m=m, tau=tau)
        metrics['largest_lyapunov_exponent'] = lle
        logger.info(f"Chaos metrics: {metrics}")
        append_log(f"Chaos metrics: {metrics}")
        if use_wandb: wandb.log(metrics)

        # 6. Regime Detection
        logger.info("Detecting regimes (HMM)...")
        hmm = train_hmm(embedded, n_components=2)
        labels, probs = predict_regimes(hmm, embedded)
        
        time_aligned = pd.to_datetime(df['time'].values[-len(labels):])
        series_aligned = normalized_signal[-len(labels):]
        
        logger.info("Detecting change points (PELT)...")
        cps = detect_changepoints_pelt(series_aligned, model_type=config['regime_detection']['cpd_model'], pen=10)
        
        # 7. Visualization
        plot_2d_attractor(embedded, save_path=f"{output_dir}/attractor_2d.png")
        if m >= 3:
            plot_3d_attractor(embedded, save_path=f"{output_dir}/attractor_3d.png")
            
        compute_and_plot_recurrence(embedded[:1000], epsilon_percentile=ep, save_path=f"{output_dir}/recurrence.png")
        plot_regime_segmentation(time_aligned, series_aligned, labels, save_path=f"{output_dir}/regimes.png")
        plot_changepoints(time_aligned, series_aligned, cps, save_path=f"{output_dir}/changepoints.png")
        
        if use_wandb and not (getattr(args, 'sweep', False) or getattr(args, 'experiment', False)):
            wandb.log({
                "attractor_2d": wandb.Image(f"{output_dir}/attractor_2d.png"),
                "regimes": wandb.Image(f"{output_dir}/regimes.png")
            })

        # 8. Dynamics Discovery (SINDy)
        logger.info("Discovering dynamics with PySINDy...")
        dt = time_aligned[1] - time_aligned[0] if len(time_aligned) > 1 else 1.0
        if hasattr(dt, 'total_seconds'):
            dt = dt.total_seconds() / (24*3600)  # converting timedelta to days if it's datetime
        elif hasattr(dt, 'days'):
            dt = float(dt.days)
        else:
            dt = float(dt)
            
        if dt <= 0:
            dt = 1.0
                
        sindy_model = fit_sparse_dynamics(embedded, t=dt, poly_degree=args.sindy_degree, threshold=args.sindy_threshold)
        equations = discover_equations(sindy_model)
        logger.info(f"Discovered SINDy equations: {equations}")
        append_log(f"Discovered SINDy equations: {equations}")

        # 9. Dynamics Discovery (Neural ODE)
        logger.info("Preparing data to train Neural ODE...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        logger.info(f"Using device: {device}")
        
        # Prepare data for Neural ODE [time, batch, dim]
        y_tensor = torch.tensor(embedded, dtype=torch.float32).unsqueeze(1).to(device)
        t_tensor = torch.tensor(np.arange(len(embedded)) * dt, dtype=torch.float32).to(device)
        
        ode_func = ODEFunc(input_dim=m, hidden_dim=config['dynamics']['neural_ode']['hidden_dim'], device=device)
        neural_ode = NeuralODE(ode_func, method=config['dynamics']['neural_ode']['solver'], device=device)
        
        trained_node, final_loss = train_neural_ode(
            neural_ode, 
            y_tensor, 
            t_tensor,
            batch_time=min(config['dynamics']['neural_ode']['batch_time'], len(embedded)),
            batch_size=config['dynamics']['neural_ode']['batch_size'],
            epochs=args.epochs,
            lr=config['dynamics']['neural_ode'].get('lr', 1e-3),
            device=device,
            use_wandb=use_wandb and not (getattr(args, 'sweep', False) or getattr(args, 'experiment', False))
        )
        append_log(f"Neural ODE training completed for tau={tau}, m={m}, epsilon_p={ep}")
        logger.info(f"Pipeline execution complete for tau={tau}, m={m}, epsilon_p={ep}.")
        
        # Log metrics to results_summary.txt
        active_terms = np.count_nonzero(sindy_model.coefficients())
        logger.info(f"SINDy active terms: {active_terms}")
        with open("results_summary.txt", "a") as f:
            f.write(f"--- Results for tau={tau}, m={m}, epsilon_p={ep} ---\n")
            f.write(f"Chaos metrics: {metrics}\n")
            f.write(f"SINDy active terms: {active_terms}\n")
            f.write(f"Final ODE Loss: {final_loss}\n\n")

if __name__ == "__main__":
    main()
