"""Constrained MEM for one trimmed DLS curve (same routine as in gfbr_mem.ipynb), kept in a module so it can run in parallel worker processes."""
import numpy as np
import casadi as ca
from scipy.optimize import nnls, curve_fit

EPS = 1e-12
NZERO = 13


def iz_from_fit(tau, g1_full, g1_zero=1e-4):
    # single-exponential fit exp(-a*tau) to the FULL noisy curve; iz = first lag time at or after the fit's g1_zero crossing
    (a,), _ = curve_fit(lambda t, a: np.exp(-a*t), tau, g1_full, p0=[1.0/tau[np.argmin(np.abs(g1_full - 0.37))]])
    return int(np.where(tau >= -np.log(g1_zero)/a)[0][0])


def mem_one(args):
    # args = (K, g1_trim, ic, iz, factor). Rows 0..ic with the data, NZERO zero rows from iz; C = factor * Cb.
    K, g1_trim, ic, iz, factor = args
    N, M = K.shape
    zeros = np.arange(iz, min(iz + NZERO, N))
    idx = np.r_[np.arange(ic + 1), zeros]
    K_ = K[idx]
    t_ = np.r_[np.asarray(g1_trim, dtype=float)[:ic + 1], np.zeros(len(zeros))]

    # feasibility boundary Cb: smallest ||K a - g1||^2 over the simplex (NNLS with a heavily weighted sum(a) = 1 row)
    A = np.vstack([K_, 1e3*np.ones((1, M))]); b = np.r_[t_, 1e3]
    a0, _ = nnls(A, b, maxiter=50000)
    Cb = float(np.sum((K_@a0 - t_)**2))
    C = factor*Cb

    # IPOPT: maximize entropy s.t. sum(a) = 1, ||K a - g1||^2 <= C
    a = ca.SX.sym("a", M)
    r = ca.mtimes(ca.DM(K_), a) - ca.DM(t_)
    solver = ca.nlpsol("s", "ipopt", {"x": a, "f": ca.sum1(a*ca.log(a + EPS)), "g": ca.vertcat(ca.sum1(a) - 1, ca.dot(r, r) - C)},
                       {"ipopt.tol": 1e-5, "ipopt.max_iter": 500, "ipopt.print_level": 0, "print_time": 0,
                        "ipopt.bound_relax_factor": 0})
    sol = solver(x0=np.full(M, 1.0/M), lbx=EPS, ubx=1.0, lbg=[0, -ca.inf], ubg=[0, 0])
    return np.array(sol["x"]).ravel(), Cb, solver.stats()["return_status"]
