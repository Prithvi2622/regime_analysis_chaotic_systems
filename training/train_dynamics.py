import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import wandb

logger = logging.getLogger(__name__)

def train_neural_ode(
    model, 
    true_y, 
    true_t,
    batch_time=10, 
    batch_size=20, 
    epochs=1000, 
    lr=0.01, 
    test_freq=10,
    device='cpu',
    use_wandb=False
):
    """
    Train the Neural ODE by matching the predicted derivatives/trajectories
    to observed trajectories using MSE loss.
    
    Args:
        model: NeuralODE instance containing the ODEFunc
        true_y: Tensor [time_len, batch_size, dim] or [time_len, dim]
        true_t: Tensor [time_len]
        batch_time: How many time steps to predict in each batch snippet.
        batch_size: How many random snippets to take per iteration.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() # Mixed precision training
    
    true_y = true_y.to(device)
    true_t = true_t.to(device)
    
    data_len = true_y.size(0)
    
    if batch_time >= data_len:
        batch_time = max(2, data_len // 2)
        logger.warning(f"batch_time reduced to {batch_time} due to data size")

    def get_batch():
        # Get random starting indices
        starts = np.random.choice(
            np.arange(data_len - batch_time, dtype=np.int64), batch_size, replace=False)
        
        starts = torch.from_numpy(starts)
        
        batch_y0 = true_y[starts]  # (batch_size, dim)
        batch_t = true_t[:batch_time]  # (batch_time,)
        
        # Build target batch: (batch_time, batch_size, dim)
        # We slice true_y at each t offset from starts
        batch_y = torch.stack([true_y[starts + i] for i in range(batch_time)], dim=0)
        
        return batch_y0, batch_t, batch_y

    for epoch in range(1, epochs + 1):
        batch_y0, batch_t, batch_y = get_batch()
        
        optimizer.zero_grad()
        
        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            # Predict forward
            pred_y = model(batch_y0, batch_t)
            # Calculate MSE loss
            loss = loss_fn(pred_y, batch_y)
            
        # Optimization step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if use_wandb:
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            
        if epoch % test_freq == 0:
            with torch.no_grad():
                model.eval()
                full_pred_y = model(true_y[0], true_t)
                full_loss = loss_fn(full_pred_y, true_y)
                logger.info(f"Epoch {epoch:04d} | Batch Loss {loss.item():.6f} | Full Trajectory Loss {full_loss.item():.6f}")
                if use_wandb:
                    wandb.log({"val_loss": full_loss.item(), "epoch": epoch})
                model.train()
                
    return model
