import numpy as np
import torch 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device=device)
SQRT_PI = torch.tensor(np.sqrt(np.pi), dtype=torch.float32, device=device)
