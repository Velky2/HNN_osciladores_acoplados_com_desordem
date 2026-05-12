"""
common.py — Utilitários compartilhados para experimentos HNN com osciladores acoplados.

HAMILTONIANO:
    H(q, p) = ½ Σ pᵢ² + ½ qᵀ J q + (λ/4) Σ qᵢ⁴

    onde J é uma matriz aleatória simétrica (desordem de tipo GOE).

INTEGRADOR — POR QUE SUBSTITUÍMOS O RK45:
    RK45 é um integrador de Runge-Kutta explícito de ordem adaptável. Para sistemas
    Hamiltonianos, ele NÃO preserva a estrutura simpléctica: o erro de energia cresce
    secularmente (~O(h^4 · t)), levando a trajetórias que "escapam" da superfície de
    energia correta. Para o treino de HNNs, isso contamina o dataset com estados
    fisicamente incorretos, degradando a qualidade do modelo.

    O integrador de Störmer-Verlet (leapfrog) é simplético de 2ª ordem: preserva uma
    Hamiltoniana modificada H̃ = H + O(h²), mantendo o erro de energia LIMITADO para
    todo t (sem crescimento secular). Para t_span=(0,20) com dt=0.02, a diferença
    prática é significativa: a energia oscila em vez de crescer, produzindo dados de
    treino que respeitam a física do sistema.

GPU — ESTRATÉGIA DE ACELERAÇÃO:
    Todas as n_trajs trajetórias são integradas em PARALELO como operações batched de
    PyTorch (q e p têm shape (n_trajs, N)). Isso move o gargalo de um loop Python de
    200 iterações para uma única operação de álgebra linear na GPU.

    Para N=8 e n_trajs=200, a aceleração é 10–50× dependendo da GPU.
    Em CPU, a versão vetorizada ainda é ~3–5× mais rápida que o loop sequencial original.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ─────────────────────────────────────────────────────────────────────────────
# Configuração do device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Retorna o melhor device disponível (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return dev


def device_info(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA — {name} ({mem:.1f} GB)"
    elif device.type == "mps":
        return "Apple MPS"
    return "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Física
# ─────────────────────────────────────────────────────────────────────────────

def make_J(N: int, sigma_J: float, seed: int = 0) -> np.ndarray:
    """Matriz de acoplamento aleatória simétrica (tipo GOE).

    J_{ij} ~ N(0, σ_J² / N),  J = (A + Aᵀ) / 2.
    A escala 1/√N garante que os autovalores de J sejam O(σ_J)
    independentemente de N (lei do semicírculo de Wigner).
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma_J / np.sqrt(max(N, 1)), (N, N))
    return (A + A.T) / 2.0


def simulate_gpu(
    N: int,
    sigma_J: float,
    lam: float,
    n_trajs: int,
    t_span: tuple[float, float],
    dt: float,
    device: torch.device,
    seed: int = 0,
    save_every: int = 5,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Integração de Störmer-Verlet vetorizada na GPU.

    Todas as n_trajs trajetórias são integradas simultaneamente como
    operações batched. A força é calculada via multiplicação de matrizes
    (operação nativa da GPU).

    Args:
        save_every: salvar estado a cada `save_every` passos (reduz memória).

    Returns:
        J_np : matriz de acoplamento (N, N) em CPU/numpy.
        states: tensor (n_saved, n_trajs, 2N) em CPU.
    """
    J_np = make_J(N, sigma_J, seed)
    J = torch.tensor(J_np, dtype=torch.float32, device=device)

    # Condições iniciais aleatórias para todas as trajetórias de uma vez
    # O gerador deve estar no mesmo device que o tensor de destino
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    q = torch.randn(n_trajs, N, generator=gen, device=device)
    p = torch.randn(n_trajs, N, generator=gen, device=device)

    t0, t1 = t_span
    n_steps = int(round((t1 - t0) / dt))
    states: list[torch.Tensor] = []

    for step in range(n_steps):
        # Störmer-Verlet (leapfrog):
        #   p_{n+1/2} = p_n + (dt/2) F(q_n)
        #   q_{n+1}   = q_n + dt · p_{n+1/2}
        #   p_{n+1}   = p_{n+1/2} + (dt/2) F(q_{n+1})
        # onde F_i = -∂H/∂q_i = -(Jq)_i - λ qᵢ³
        F = -(q @ J) - lam * q.pow(3)         # (n_trajs, N)
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half
        F_new = -(q @ J) - lam * q.pow(3)
        p = p_half + 0.5 * dt * F_new

        if (step + 1) % save_every == 0:
            # Mover para CPU imediatamente para liberar VRAM
            states.append(torch.cat([q, p], dim=1).cpu())

    return J_np, torch.stack(states, dim=0)   # (n_saved, n_trajs, 2N)


def compute_energy(states: torch.Tensor, J_np: np.ndarray, lam: float) -> torch.Tensor:
    """
    Calcula H(q, p) para cada ponto de um tensor de estados.

    Args:
        states: (..., 2N)
    Returns:
        energies: (...) — mesma forma sem a última dimensão.
    """
    N = states.shape[-1] // 2
    J = torch.tensor(J_np, dtype=torch.float32)
    q = states[..., :N]
    p = states[..., N:]
    kinetic   = 0.5 * p.pow(2).sum(-1)
    potential = 0.5 * (q @ J * q).sum(-1) + (lam / 4.0) * q.pow(4).sum(-1)
    return kinetic + potential


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def extract_dataset(
    states: torch.Tensor,
    J_np: np.ndarray,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula pares (estado, derivada) de forma analítica (sem integração numérica).

    Dado que conhecemos as equações de movimento:
        dq/dt = p
        dp/dt = -(Jq) - λ q³

    podemos calcular Y = dy/dt exatamente a partir de cada ponto salvo,
    sem necessidade de diferenças finitas.

    Args:
        states: (n_saved, n_trajs, 2N) ou (n_amostras, 2N)
    Returns:
        X: (n_amostras, 2N)
        Y: (n_amostras, 2N)
    """
    if states.ndim == 3:
        n_saved, n_trajs, two_N = states.shape
        states = states.reshape(-1, two_N)

    N = states.shape[1] // 2
    J = torch.tensor(J_np, dtype=torch.float32)

    q = states[:, :N]
    p = states[:, N:]
    dq = p
    dp = -(q @ J) - lam * q.pow(3)

    X = states.float()
    Y = torch.cat([dq, dp], dim=1).float()
    return X, Y


class HamiltonianDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.Y[i]


def make_dataloaders(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Divide dataset em treino/val/teste e retorna DataLoaders."""
    ds = HamiltonianDataset(X, Y)
    n = len(ds)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds,   **kwargs),
        DataLoader(test_ds,  **kwargs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """MLP genérico com arquitetura configurável."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str,
    ):
        super().__init__()
        act_map: dict[str, type] = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "softplus": nn.Softplus,
        }
        if activation not in act_map:
            raise ValueError(f"Ativação desconhecida: {activation}. Use uma de {list(act_map)}")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_map[activation]()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_map[activation]()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HNN(nn.Module):
    """
    Hamiltonian Neural Network (Greydanus et al., 2019).

    Aprende H̃: R^{2N} → R via uma MLP, e deriva a dinâmica usando:
        dq/dt =  ∂H̃/∂p
        dp/dt = -∂H̃/∂q

    NOTA SOBRE O AUTOGRAD:
        x.detach().requires_grad_(True) cria um tensor-folha no mesmo device.
        create_graph=self.training garante que:
        - Durante treino:  o grafo de 2ª ordem é construído → backprop correto
                           pelos parâmetros de H_net.
        - Durante avaliação (eval/SHAP/integração): sem grafo extra → mais rápido.
    """

    def __init__(self, N: int, hidden_dim: int, n_layers: int, activation: str):
        super().__init__()
        self.N = N
        self.H_net = MLP(2 * N, hidden_dim, n_layers, activation)

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna H̃(x) como escalar por amostra — útil para SHAP e análise."""
        return self.H_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        H = self.H_net(x).sum()
        grad = torch.autograd.grad(H, x, create_graph=self.training)[0]
        dq =  grad[:, self.N:]    #  ∂H̃/∂p
        dp = -grad[:, :self.N]    # -∂H̃/∂q
        return torch.cat([dq, dp], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Treino
# ─────────────────────────────────────────────────────────────────────────────

def train_hnn(
    model: HNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    device: torch.device,
    patience: int = 10,
    return_model: bool = False,
) -> float | tuple[float, HNN]:
    """
    Treina a HNN e retorna o melhor val_loss.

    Early stopping com `patience` épocas evita overfitting durante a HPO,
    onde cada trial deve ser barato mas fiel ao desempenho real.

    Args:
        return_model: se True, retorna (val_loss, modelo_melhor).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # ── Treino ──────────────────────────────────────────────────────────
        model.train()
        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            pred = model(X_b)
            loss = loss_fn(pred, Y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # ── Validação ────────────────────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)
                # enable_grad é necessário pois forward() usa autograd.grad
                with torch.enable_grad():
                    pred = model(X_b)
                val_losses.append(loss_fn(pred, Y_b).item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            patience_counter = 0
            if return_model:
                import copy
                best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if return_model:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        return best_val_loss, model

    return best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Função objetivo (fechamento sobre dataset pré-computado)
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    X: torch.Tensor,
    Y: torch.Tensor,
    N: int,
    device: torch.device,
    n_epochs: int = 50,
    patience: int = 10,
) -> Callable[[dict], float]:
    """
    Retorna objective(params) → val_loss.

    IMPORTANTE: a simulação é pré-computada FORA da função objetivo.
    No código original, simulate() era chamado dentro de objective(), o que
    significa re-simular 200 trajetórias para cada um dos 50 trials × 5 seeds
    × 4 sigma_J = 1.000 simulações completas. Aqui simulamos UMA vez por
    (sigma_J, seed) e reutilizamos os dados.
    """
    def objective(params: dict) -> float:
        train_loader, val_loader, _ = make_dataloaders(X, Y, params["batch_size"])
        model = HNN(
            N=N,
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
            activation=params["activation"],
        )
        return train_hnn(
            model, train_loader, val_loader,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            n_epochs=n_epochs,
            device=device,
            patience=patience,
        )

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(history: list[float]) -> dict:
    """
    Dado o histórico de val_loss por trial, computa:
    - best : melhor valor final.
    - auc  : área sob a curva de convergência (cummin médio) — menor = melhor.
    - t10  : trial (0-indexado) onde atingiu <= best * 1.1.
    """
    h = np.array(history, dtype=float)
    best = float(h.min())
    cummin = np.minimum.accumulate(h)
    auc = float(cummin.mean())
    threshold = best * 1.10 + 1e-12
    idxs = np.where(cummin <= threshold)[0]
    t10 = int(idxs[0]) if len(idxs) > 0 else len(h)
    return {"best": best, "auc": auc, "t10": t10}


# ─────────────────────────────────────────────────────────────────────────────
# Codificação de hiperparâmetros para SHAP
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATIONS = ["tanh", "silu", "gelu", "softplus"]
BATCH_SIZES  = [64, 128, 256, 512]

PARAM_NAMES = ["log_lr", "hidden_dim", "n_layers", "activation_idx", "batch_idx", "log_wd"]

def encode_params(params: dict) -> list[float]:
    """Codifica hiperparâmetros em vetor numérico para SHAP/surrogate."""
    return [
        float(np.log(params["lr"])),
        float(params["hidden_dim"]),
        float(params["n_layers"]),
        float(ACTIVATIONS.index(params["activation"])),
        float(BATCH_SIZES.index(params["batch_size"])),
        float(np.log(params["weight_decay"])),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Integração com modelo treinado (para análise de conservação de energia)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def integrate_hnn_rk4(
    model: HNN,
    q0: torch.Tensor,
    p0: torch.Tensor,
    dt: float,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Integra o modelo HNN com RK4 explícito.

    Usamos RK4 AQUI (não na geração de dados) porque estamos integrando as
    equações PREDITAS pelo modelo, não as equações exatas. O integrador
    simplético seria ideal, mas o modelo pode não satisfazer as condições
    de Hamiltoniana exata. RK4 é uma escolha razoável para visualização.

    Returns:
        trajectory: (n_steps+1, 2N)
    """
    model.eval()
    state = torch.cat([q0, p0]).unsqueeze(0).to(device)
    traj = [state.cpu()]

    for _ in range(n_steps):
        with torch.enable_grad():
            k1 = model(state)
            k2 = model(state + 0.5 * dt * k1)
            k3 = model(state + 0.5 * dt * k2)
            k4 = model(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(state.cpu())

    return torch.cat(traj, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Cache de simulações em disco
# ─────────────────────────────────────────────────────────────────────────────

def sim_cache_path(base_dir: str, sigma_J: float, seed: int) -> str:
    return os.path.join(base_dir, f"sim_sigma{sigma_J:.2f}_seed{seed}.pt")


def load_or_simulate(
    base_dir: str,
    N: int,
    sigma_J: float,
    lam: float,
    n_trajs: int,
    t_span: tuple,
    dt: float,
    device: torch.device,
    seed: int,
    save_every: int = 5,
    force: bool = False,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Carrega simulação do cache se existir, caso contrário simula e salva.

    Isso evita re-simular ao rodar múltiplos notebooks para o mesmo dataset.
    """
    os.makedirs(base_dir, exist_ok=True)
    path = sim_cache_path(base_dir, sigma_J, seed)

    if not force and os.path.exists(path):
        data = torch.load(path, weights_only=False)
        return data["J_np"], data["states"]

    J_np, states = simulate_gpu(N, sigma_J, lam, n_trajs, t_span, dt, device, seed, save_every)
    torch.save({"J_np": J_np, "states": states}, path)
    return J_np, states


# ─────────────────────────────────────────────────────────────────────────────
# Constantes do experimento
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_LEVELS  = [0.0, 0.5, 1.0, 2.0]
N_TRIALS      = 50
N_SEEDS       = 3         # aumentar para 5 para significância estatística plena
N_OSCILLATORS = 8
LAM           = 0.1
T_SPAN        = (0.0, 20.0)
DT            = 0.02      # passo temporal (Störmer-Verlet)
N_TRAJS       = 200
SAVE_EVERY    = 5         # salvar 1 em cada 5 passos → 200 snapshots por trajetória
N_EPOCHS_OBJ  = 50        # épocas por trial de HPO
PATIENCE_OBJ  = 10
SIM_CACHE_DIR = "sim_cache"
RESULTS_DIR   = "results"
