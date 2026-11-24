"""SHIFT 2D particle simulator (deterministic, unit mass, fixed dt).

State: float64 array, shape (N, 4), columns [x, y, vx, vy].
Trajectory: shape (T+1, N, 4); index 0 is the initial state.
A force function maps state (N, 4) -> force array (N, 2).
"""
import numpy as np

MASS = 1.0
X, Y, VX, VY = 0, 1, 2, 3


def init_state(n, seed, pos_range=1.0, vel_range=1.0):
    """Random initial state (N, 4); same seed -> identical result."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-pos_range, pos_range, size=(n, 2))
    vel = rng.uniform(-vel_range, vel_range, size=(n, 2))
    return np.hstack([pos, vel])


def constant_force(fx=0.0, fy=0.0):
    """Return a force function giving every particle the same force (N, 2)."""
    f = np.array([fx, fy], dtype=float)
    return lambda state: np.tile(f, (state.shape[0], 1))


def restoring_force(k, center_x=0.0, center_y=0.0):
    """Harmonic force factory: F = -k * (position - center), unit mass => a = -k*(p - c).

    Depends on the particle's CURRENT position only. k is a parameter of the force,
    so the integrator (`step`) is unchanged.
    """
    c = np.array([center_x, center_y], dtype=float)
    return lambda state: -k * (state[:, 0:2] - c)


def acceleration(state, force_fn, mass=MASS):
    """a = F / m, shape (N, 2)."""
    return force_fn(state) / mass


def step(state, force_fn, dt, mass=MASS):
    """One semi-implicit Euler step: v' = v + a*dt; p' = p + v'*dt."""
    a = acceleration(state, force_fn, mass)
    v_next = state[:, 2:4] + a * dt
    p_next = state[:, 0:2] + v_next * dt
    return np.hstack([p_next, v_next])


def run(state0, force_fn, dt, n_steps, mass=MASS):
    """Simulate n_steps; return trajectory of shape (n_steps+1, N, 4)."""
    traj = np.empty((n_steps + 1,) + state0.shape)
    traj[0] = state0
    for t in range(n_steps):
        traj[t + 1] = step(traj[t], force_fn, dt, mass)
    return traj


def export_transitions(traj, force_fn, mass=MASS):
    """Flatten a trajectory into supervised transition data.

    Returns (current, next, accel):
      current (T*N, 4), next (T*N, 4), accel (T*N, 2) true a = F/m at current state.
    """
    cur = traj[:-1].reshape(-1, 4)
    nxt = traj[1:].reshape(-1, 4)
    acc = acceleration(cur, force_fn, mass)
    return cur, nxt, acc


def generate_dataset(force_fn, n_traj, n_steps, dt, seed, pos_range=1.0, vel_range=1.0):
    """Reproducibly generate many trajectories and their transition data.

    Particles do not interact, so each of the n_traj particles is an independent
    trajectory with its own random initial position/velocity.
    Returns (traj, (current, next, accel)); traj (n_steps+1, n_traj, 4).
    """
    s0 = init_state(n_traj, seed, pos_range, vel_range)
    traj = run(s0, force_fn, dt, n_steps)
    return traj, export_transitions(traj, force_fn)