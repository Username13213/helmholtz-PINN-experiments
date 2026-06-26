import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

k = 100

# Model ======================================================================
class SIREN(nn.Module):
    def __init__(self, layers, w0_init=3.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0_init, dtype=torch.float32))
        self.lb, self.ub = 0.0, 1.0

        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i+1])
            if i == 0:
                std = 1.0/np.sqrt(layers[i])
            else:
                std = np.sqrt(1.0/layers[i])/w0_init
            nn.init.uniform_(lin.weight, -std, std)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)

    def forward(self, x):
        H = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        H = self.w0*H
        for lin in self.linears[:-1]:
            H = torch.sin(lin(H))
        return self.linears[-1](H)

def pde(model, x, f):
    y = model(x)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]
    return y_xx + k**2*y - f

def loss_fn(model, x, x_bc, y_bc, f, lambda_bc=500.0):
    res = pde(model, x, f)
    loss_pde = torch.mean(res**2)
    y_bc_pred = model(x_bc)
    loss_bc = torch.mean((y_bc - y_bc_pred)**2)
    return loss_pde + lambda_bc*loss_bc

# Computed Solution ======================================================================
Lx = 1.0
Nx = 5000

dx = Lx/(Nx - 1)
x = np.linspace(0, Lx, Nx, dtype=np.float32)


sigma = 0.02
f = np.exp(-((x - 0.5)**2)/(2*sigma**2))

rows, cols, data = [], [], []
b = f.copy()

for i in range(Nx):
    if i == 0 or i == Nx - 1:                      
        rows.append(i); cols.append(i); data.append(1.0)
        b[i] = 0.0
    else:                                           
        rows.append(i); cols.append(i); data.append(-2/dx**2 + k**2)
        rows.append(i); cols.append(i + 1); data.append(1/dx**2)
        rows.append(i); cols.append(i - 1); data.append(1/dx**2)
 
A = sp.csr_matrix((data, (rows, cols)), shape=(Nx, Nx))
y_computed = spla.spsolve(A, b)

# Training data ======================================================================
x_train = torch.from_numpy(x).reshape(-1, 1).to(device).requires_grad_(True)
f_train = torch.from_numpy(f).reshape(-1, 1).to(device).requires_grad_(True)

x_bc = torch.tensor([[0.0], [1.0]]).to(device)
y_bc = torch.tensor([[0.0], [0.0]]).to(device)

# Training ======================================================================
model = SIREN(layers=[1, 128, 128, 128, 128, 128, 1], w0_init=float(k)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/1000))

epochs = 100_000

pbar = tqdm(range(epochs), desc="Adam")

for epoch in pbar:
    optimizer.zero_grad()

    loss_value = loss_fn(model, x_train, x_bc, y_bc, f_train)

    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    pbar.set_postfix({
        "loss": f"{loss_value.item():.4e}",
        "w0": f"{model.w0.item():.3f}"
    })

optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, history_size=50, line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_fn(model, x_train, x_bc, y_bc, f_train)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)

# Plot ======================================================================
x_test = torch.from_numpy(x).float().reshape(-1, 1).to(device)
with torch.no_grad():
    y_pred = model(x_test)

x_plot = x_test.detach().cpu().numpy()
y_pred = y_pred.detach().cpu().numpy()

plt.plot(x_plot, y_computed, 'b-', label="Solution analytique")
plt.plot(x_plot, y_pred, 'r--', label="Solution PINN")
plt.legend()
plt.show()
