import yaml
import logging
import pandas as pd
import numpy as np
import torch
import warnings
import wandb
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Import our modules
from utils.data_loader import load_data
from utils.preprocessing import compute_log_returns, normalize_series, denoise_signal
from analysis.takens_embedding import compute_delay_mutual_information, false_nearest_neighbors, takens_embedding
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

def main():
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    use_wandb = config['wandb']['mode'] == 'online'
    if use_wandb:
        wandb.init(project=config['wandb']['project'], config=config)
        
    # Generate data if not exists
    raw_path = config['data']['raw_path']
    if not Path(raw_path).exists():
        create_synthetic_data(raw_path)

    # 2. Data Loading
    df = load_data(raw_path, time_col=config['data']['time_col'])
    
    # 3. Preprocessing
    df = compute_log_returns(df, price_col=config['data']['target_col'])
    signal = df['returns'].values
    
    # Denoise
    meth = config['preprocessing']['denoise_method']
    logger.info(f"Denoising with {meth}")
    denoised_signal = denoise_signal(signal, method=meth, window_size=config['preprocessing']['window_size'])
    normalized_signal = normalize_series(denoised_signal)
    
    # 4. Phase Space Reconstruction
    tau_max = config['phase_space']['tau_max']
    m_max = config['phase_space']['m_max']
    
    tau = compute_delay_mutual_information(normalized_signal, tau_max=tau_max)
    m = false_nearest_neighbors(normalized_signal, tau=tau, m_max=m_max)
    logger.info(f"Estimated parameters: delay (tau) = {tau}, embedding dimension (m) = {m}")
    
    embedded = takens_embedding(normalized_signal, tau=tau, m=m)
    logger.info(f"Phase space reconstructed. Shape: {embedded.shape}")
    
    # 5. Chaos Metrics
    logger.info("Computing chaos metrics...")
    metrics = get_chaos_metrics(normalized_signal, m=m, from_pyunicorn=False)
    lle = compute_largest_lyapunov_exponent(normalized_signal, m=m, tau=tau)
    metrics['largest_lyapunov_exponent'] = lle
    logger.info(f"Chaos metrics: {metrics}")
    if use_wandb: wandb.log(metrics)

    # 6. Regime Detection
    logger.info("Detecting regimes (HMM)...")
    hmm = train_hmm(embedded, n_components=config['regime_detection']['n_components'])
    labels, probs = predict_regimes(hmm, embedded)
    
    time_aligned = df['time'].values[-len(labels):]
    series_aligned = normalized_signal[-len(labels):]
    
    logger.info("Detecting change points (PELT)...")
    cps = detect_changepoints_pelt(series_aligned, model_type=config['regime_detection']['cpd_model'])
    
    # 7. Visualization
    Path("outputs").mkdir(exist_ok=True)
    plot_2d_attractor(embedded, save_path="outputs/attractor_2d.png")
    if m >= 3:
        plot_3d_attractor(embedded, save_path="outputs/attractor_3d.png")
        
    compute_and_plot_recurrence(embedded[:1000], save_path="outputs/recurrence.png") # limit size for speed
    plot_regime_segmentation(time_aligned, series_aligned, labels, save_path="outputs/regimes.png")
    plot_changepoints(time_aligned, series_aligned, cps, save_path="outputs/changepoints.png")
    
    if use_wandb:
        wandb.log({
            "attractor_2d": wandb.Image("outputs/attractor_2d.png"),
            "regimes": wandb.Image("outputs/regimes.png")
        })

    # 8. Dynamics Discovery (SINDy)
    logger.info("Discovering dynamics with PySINDy...")
    dt = time_aligned[1] - time_aligned[0] if len(time_aligned) > 1 else 1.0
    sindy_model = fit_sparse_dynamics(embedded, t=dt, 
                                      poly_degree=config['dynamics']['sindy']['polynomial_degree'])
    equations = discover_equations(sindy_model)
    logger.info(f"Discovered SINDy equations: {equations}")

    # 9. Dynamics Discovery (Neural ODE)
    logger.info("Training Neural ODE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Prepare data for Neural ODE [time, batch, dim]
    y_tensor = torch.tensor(embedded, dtype=torch.float32).unsqueeze(1) # (T, 1, m)
    t_tensor = torch.tensor(np.arange(len(embedded)) * dt, dtype=torch.float32) # (T,)
    
    ode_func = ODEFunc(input_dim=m, hidden_dim=config['dynamics']['neural_ode']['hidden_dim'])
    neural_ode = NeuralODE(ode_func, method=config['dynamics']['neural_ode']['solver'])
    
    trained_node = train_neural_ode(
        neural_ode, 
        y_tensor, 
        t_tensor,
        batch_time=min(config['dynamics']['neural_ode']['batch_time'], len(embedded)),
        batch_size=config['dynamics']['neural_ode']['batch_size'],
        epochs=min(20, config['dynamics']['neural_ode']['epochs']), # override config with short epochs for quick test
        lr=config['dynamics']['neural_ode']['lr'],
        device=device,
        use_wandb=use_wandb
    )
    logger.info("Pipeline execution complete.")

if __name__ == "__main__":
    main()
