import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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

    def p(self, x): return self.forward(x)[:, 0:1]
    def q(self, x): return self.forward(x)[:, 1:2]

def analytical_solution(x_np, k, n_d, x_a, x_b):
    ka = k*x_a
    kb = k*x_b
    nka = n_d*ka
    nkb = n_d*kb

    M = np.array([
        [np.exp(-1j*ka), -np.exp(1j*nka), -np.exp(-1j*nka), 0],
        [-1j*k*np.exp(-1j*ka), -1j*n_d*k*np.exp(1j*nka), 1j*n_d*k*np.exp(-1j*nka), 0],
        [0, np.exp(1j*nkb), np.exp(-1j*nkb), -np.exp(1j*kb) ],
        [0, 1j*n_d*k*np.exp(1j*nkb), -1j*n_d*k*np.exp(-1j*nkb), -1j*k*np.exp(1j*kb)],
    ], dtype=complex)

    rhs = np.array([-np.exp(1j*ka), -1j*k*np.exp(1j*ka), 0, 0], dtype=complex)

    R, C, D, T = np.linalg.solve(M, rhs)

    u = np.zeros(len(x_np), dtype=complex)
    for i, xi in enumerate(x_np):
        if xi < x_a:
            u[i] = np.exp(1j*k*xi) + R*np.exp(-1j*k*xi)
        elif xi <= x_b:
            u[i] = C*np.exp(1j*n_d*k*xi) + D*np.exp(-1j*n_d*k*xi)
        else:
            u[i] = T*np.exp(1j*k*xi)

    return u, R, T

def get_n2(x_t, x_a, x_b, n_d):
    x_np = x_t.detach().cpu().numpy().flatten()
    n2   = np.ones_like(x_np)
    n2[(x_np >= x_a) & (x_np <= x_b)] = n_d**2
    return torch.tensor(n2, dtype=torch.float32, device=x_t.device).reshape_as(x_t)

def get_grad1(u, x):
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def get_grad_2(u, x):
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx

def compute_pde_residuals(model, x, k, x_a, x_b, n_d):
    p, q   = model.p(x), model.q(x)
    p_xx   = get_grad_2(p, x)
    q_xx   = get_grad_2(q, x)
    n2     = get_n2(x, x_a, x_b, n_d)
    contr  = 1.0 - n2                        

    src_p  = k**2*contr*torch.cos(k*x)  
    src_q  = k**2*contr*torch.sin(k*x)  

    res_p  = p_xx + k**2*n2*p - src_p
    res_q  = q_xx + k**2*n2*q - src_q

    return res_p, res_q

def compute_bc_loss(model, x_l, x_r, k):
    p_l, q_l   = model.p(x_l), model.q(x_l)
    p_lx, q_lx = get_grad1(p_l, x_l), get_grad1(q_l, x_l)
    loss_l = (p_lx - k*q_l)**2 + (q_lx + k*p_l)**2

    p_r, q_r   = model.p(x_r), model.q(x_r)
    p_rx, q_rx = get_grad1(p_r, x_r), get_grad1(q_r, x_r)
    loss_r = (p_rx + k*q_r)**2 + (q_rx - k*p_r)**2

    return loss_l.mean() + loss_r.mean()

def compute_total_loss(model, x_coll, x_l, x_r, k, x_a, x_b, n_d, lambda_bc=20.0):
    res_p, res_q = compute_pde_residuals(model, x_coll, k, x_a, x_b, n_d)
    l_pde = torch.mean(res_p**2) + torch.mean(res_q**2)
    l_bc  = compute_bc_loss(model, x_l, x_r, k)

    return l_pde + lambda_bc*l_bc, l_pde, l_bc

def make_training_data(Nx, L):
    x_np = np.linspace(0, L, Nx, dtype=np.float32)
    x_coll = torch.from_numpy(x_np).reshape(-1, 1).to(device).requires_grad_(True)
    x_l = torch.tensor([[0.0]], device=device).requires_grad_(True)
    x_r = torch.tensor([[L]],   device=device, dtype=torch.float32).requires_grad_(True)
    return x_np, x_coll, x_l, x_r

def run_adam_training(model, x_coll, x_l, x_r, k, x_a, x_b, n_d, epochs=100_000, lr=1e-3, lambda_bc=20.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/10_000))
    hist = {"total": [], "pde": [], "bc": []}

    pbar = tqdm(range(epochs), desc="Adam")
    for ep in pbar:
        optimizer.zero_grad()
        loss, lp, lb = compute_total_loss(model, x_coll, x_l, x_r, k, x_a, x_b, n_d, lambda_bc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        hist["total"].append(loss.item())
        hist["pde"].append(lp.item())
        hist["bc"].append(lb.item())

        pbar.set_postfix({"loss": f"{loss:.3e}", "pde": f"{lp:.3e}", "bc": f"{lb:.3e}", "w0": f"{model.w0.item():.2f}"})

    return hist

def run_lbfgs_training(model, x_coll, x_l, x_r, k, x_a, x_b, n_d, max_iter=10_000, lambda_bc=20.0):
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, history_size=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss, _, _ = compute_total_loss(model, x_coll, x_l, x_r, k, x_a, x_b, n_d, lambda_bc)
        loss.backward()
        return loss

    opt.step(closure)

def get_final_field(model, x_np, k):
    x_test = torch.from_numpy(x_np).reshape(-1, 1).to(device)

    with torch.no_grad():
        out = model(x_test).cpu().numpy()

    p_pred = out[:, 0]
    q_pred = out[:, 1]
    u_scat_pred = p_pred + 1j*q_pred
    u_total_pred = np.exp(1j*k*x_np) + u_scat_pred
    return p_pred, q_pred, u_scat_pred, u_total_pred

def plot_results(x_np, u_exact, u_total_pred, p_pred, q_pred, hist, k, n_d, x_a, x_b):
    u_inc_np = np.exp(1j*k*x_np)
    u_scat_exact = u_exact - u_inc_np
    slab_kw = dict(alpha=0.12, color='green')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"1D Scattering — dielectric slab — k={k}, n_d={n_d}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(x_np, u_exact.real, 'b-',  lw=1.5, label="Analytical")
    ax.plot(x_np, u_total_pred.real, 'r--', lw=1.2, label="PINN")
    ax.axvspan(x_a, x_b, **slab_kw, label=f"Slab n={n_d}")
    ax.set_title("Re(u_total)"); ax.set_xlabel("x"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x_np, np.abs(u_exact), 'b-',  lw=1.5, label="Analytical")
    ax.plot(x_np, np.abs(u_total_pred), 'r--', lw=1.2, label="PINN")
    ax.axvspan(x_a, x_b, **slab_kw)
    ax.set_title("|u_total|"); ax.set_xlabel("x"); ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(x_np, u_scat_exact.real, 'b-',  lw=1.5, label="Analytical")
    ax.plot(x_np, p_pred, 'r--', lw=1.2, label="PINN  p(x)")
    ax.axvspan(x_a, x_b, **slab_kw)
    ax.set_title("Re(u_scat)"); ax.set_xlabel("x"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(x_np, u_scat_exact.imag, 'b-',  lw=1.5, label="Analytical")
    ax.plot(x_np, q_pred, 'r--', lw=1.2, label="PINN  q(x)")
    ax.axvspan(x_a, x_b, **slab_kw)
    ax.set_title("Im(u_scat)"); ax.set_xlabel("x"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(x_np, np.abs(u_total_pred - u_exact) + 1e-16, 'g-', lw=1)
    ax.axvspan(x_a, x_b, **slab_kw)
    rel_err = np.linalg.norm(u_total_pred - u_exact)/np.linalg.norm(u_exact)
    ax.set_title(f"Absolute error — L2 rel. = {rel_err:.2e}"); ax.set_xlabel("x")

    ax = axes[1, 2]
    ax.semilogy(hist["total"], 'k', lw=0.7, label="Total")
    ax.semilogy(hist["pde"], 'b', lw=0.7, label="PDE")
    ax.semilogy(hist["bc"], 'r', lw=0.7, label="BC")
    ax.set_title("Loss history"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"scattering_1d_result_k_{k}.png", dpi=150, bbox_inches='tight')


# Main loop ======================================================================
k_list = [5, 10, 15, 20, 30, 40, 50]
n_d = 1.5
x_a, x_b, L = 0.35, 0.65, 1.0
Nx = 2000 

x_np, x_coll, x_l, x_r = make_training_data(Nx, L)

for k in k_list:
    u_exact, _, _ = analytical_solution(x_np, k, n_d, x_a, x_b)

    model = SIREN(layers=[1, 256, 256, 256, 256, 2], w0_init=3.0, w0_start=float(k)).to(device)

    hist = run_adam_training(model, x_coll, x_l, x_r, k, x_a, x_b, n_d)
    run_lbfgs_training(model, x_coll, x_l, x_r, k, x_a, x_b, n_d)

    p_pred, q_pred, _, u_total_pred = get_final_field(model, x_np, k)
    plot_results(x_np, u_exact, u_total_pred, p_pred, q_pred, hist, k, n_d, x_a, x_b)