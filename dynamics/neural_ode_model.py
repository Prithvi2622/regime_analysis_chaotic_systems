import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

class ODEFunc(nn.Module):
    """
    Neural ODE function parameterizing the derivative dX/dt.
    Network architecture: Linear -> Tanh -> Linear -> Tanh -> Linear
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, device: str = None):
        super(ODEFunc, self).__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        """
        Calculates the derivative at time t and state x.
        """
        return self.net(x)

class NeuralODE(nn.Module):
    """
    Neural ODE wrapper.
    """
    def __init__(self, func: ODEFunc, method: str = 'rk4', device: str = None):
        super(NeuralODE, self).__init__()
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq is required to use NeuralODE.")
        self.func = func
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.func.to(self.device)
        
    def forward(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Simulate trajectories.
        
        Args:
            x0: Initial state [batch_size, input_dim] or [input_dim]
            t: Time evaluation points [T]
            
        Returns:
            Trajectories [T, batch_size, input_dim] or [T, input_dim]
        """
        # odeint expects func(t, x), initial states, and evaluation times
        out = odeint(self.func, x0, t, method=self.method)
        return out
