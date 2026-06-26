import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model ======================================================================
class SIREN(nn.Module):
    def __init__(self, layers, w0_init=3.0, w0_start=None):  
        super().__init__()
        w0_start = w0_start if w0_start is not None else w0_init
        self.w0 = nn.Parameter(torch.tensor(w0_start, dtype=torch.float32))
        self.lb, self.ub = 0.0, 1.0
        self.linears = nn.ModuleList()

        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i+1])
            std = 1.0/np.sqrt(layers[i]) if i == 0 else np.sqrt(1.0/layers[i])/w0_init
            nn.init.uniform_(lin.weight, -std, std)
            nn.init.zeros_(lin.bias)
            self.linears.append(lin)

    def forward(self, x):
        H = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        H = self.w0 * H
        for lin in self.linears[:-1]:
            H = torch.sin(lin(H))
        return self.linears[-1](H)
    
def pde(model, xy, f):
    u = model(xy)

    grads = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    return u_xx + u_yy + k**2*u - f

def compute_loss(model, xy, xy_bc, u_bc, f, lambda_bc=20.0):
    res = pde(model, xy, f)
    loss_pde = torch.mean(res**2)
    u_bc_pred = model(xy_bc)
    loss_bc = torch.mean((u_bc - u_bc_pred)**2)
    return loss_pde + lambda_bc*loss_bc

def get_idx(i, j, Ny):
    return i*Ny + j

k = 20

# Computed Solution ======================================================================
Lx, Ly = 1, 1
Nx, Ny = 500, 500
edges_x = [0, Nx - 1]
edges_y = [0, Ny - 1]

dx, dy = Lx/(Nx - 1), Ly/(Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")

sigma = 0.05
f = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2)/(2*sigma**2))

N = Nx*Ny

rows, cols, data = [], [], []
b = f.flatten().copy()

for i in range(Nx):
    for j in range(Ny):
        p = get_idx(i, j, Ny)

        if i in edges_x or j in edges_y:
            rows.append(p); cols.append(p); data.append(1.0)
            b[p] = 0.0

        else:
            rows.append(p); cols.append(p); data.append(-2/dx**2 - 2/dy**2 + k**2) 
            rows.append(p); cols.append(get_idx(i + 1, j, Ny)); data.append(1/dx**2) 
            rows.append(p); cols.append(get_idx(i - 1, j, Ny)); data.append(1/dx**2) 
            rows.append(p); cols.append(get_idx(i, j + 1, Ny)); data.append(1/dy**2) 
            rows.append(p); cols.append(get_idx(i, j - 1, Ny)); data.append(1/dy**2) 

A = sp.csr_matrix((data, (rows, cols)), shape = (N, N))
u = spla.spsolve(A, b)
computed_result = u.reshape(Nx, Ny)

# Training data ======================================================================
XY = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
xy_train = torch.from_numpy(XY).to(device).requires_grad_(True)
f_train  = torch.from_numpy(f.flatten().astype(np.float32)).reshape(-1, 1).to(device)

bc_x0 = np.stack([np.zeros(Ny), y], axis=1)
bc_x1 = np.stack([np.ones(Ny), y], axis=1)
bc_y0 = np.stack([x, np.zeros(Nx)], axis=1)
bc_y1 = np.stack([x, np.ones(Nx)],  axis=1)

xy_bc_np = np.concatenate([bc_x0, bc_x1, bc_y0, bc_y1], axis=0).astype(np.float32)
u_bc_np  = np.zeros((xy_bc_np.shape[0], 1), dtype=np.float32)

xy_bc = torch.from_numpy(xy_bc_np).to(device)
u_bc  = torch.from_numpy(u_bc_np).to(device)

# Training ======================================================================
model = SIREN(layers=[2, 128, 128, 128, 128, 128, 1], w0_init=3.0, w0_start=float(k)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/1000))

epochs = 10_000

N_train = Nx*Ny
batch_size = 4000

loss_history = []   
pbar = tqdm(range(epochs), desc="Adam")

for epoch in pbar:
    optimizer.zero_grad()

    idx = torch.randperm(N_train, device=device)[:batch_size]
    xy_batch = xy_train[idx]        
    f_batch  = f_train[idx]

    loss_value = compute_loss(model, xy_batch, xy_bc, u_bc, f_batch)

    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    loss_history.append(loss_value.item())   


    pbar.set_postfix({
        "loss": f"{loss_value.item():.4e}",
        "w0": f"{model.w0.item():.3f}"
    })

def closure():
    optimizer_lbfgs.zero_grad()
    idx = torch.randperm(N_train, device=device)[:batch_size]
    loss = compute_loss(model, xy_train[idx], xy_bc, u_bc, f_train[idx])
    loss.backward()
    return loss

optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, history_size=50, line_search_fn="strong_wolfe")
optimizer_lbfgs.step(closure)

# Plot ======================================================================
XY_test = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
xy_test = torch.from_numpy(XY_test).to(device)

with torch.no_grad():
    u_pred = model(xy_test).cpu().numpy().reshape(Nx, Ny)

error = (computed_result - u_pred)**2

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f"Helmholtz 2D — k={k}", fontsize=14)

im0 = axes[0, 0].pcolormesh(X, Y, computed_result, cmap='RdBu', shading='auto')
axes[0, 0].set_title("Numerical solution")
axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].pcolormesh(X, Y, u_pred, cmap='RdBu', shading='auto')
axes[0, 1].set_title("PINN solution")
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].pcolormesh(X, Y, error, cmap='hot_r', shading='auto')
axes[1, 0].set_title("Error $(u_{FD} - u_{PINN})^2$")
axes[1, 0].set_ylabel("y")
plt.colorbar(im2, ax=axes[1, 0])

rel_err = np.sqrt(error.sum())/np.linalg.norm(computed_result)
axes[1, 0].set_xlabel(f"relative L2 error = {rel_err:.3e}")

axes[1, 1].semilogy(loss_history, 'k-', lw=0.7)
axes[1, 1].set_title("Loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()