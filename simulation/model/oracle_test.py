"""Oracle test: feed TRUE networks to MLE, verify parameter recovery."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'model'))
from multilayer_network_mle_integrated import estimate_single_window, row_normalize

np.random.seed(42)

N = 5
T = 1000

W_vol_raw = np.array([[0,1,1,0,1],[1,0,1,1,0],[1,1,0,1,1],[0,1,1,0,1],[1,0,1,1,0]], dtype=float)
W_tail_raw = np.array([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]], dtype=float)
W_ret_raw = np.array([[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,1],[0,1,0,0,0],[0,0,1,0,0]], dtype=float)

W_vol = row_normalize(W_vol_raw)
W_tail = row_normalize(W_tail_raw)
W_ret = row_normalize(W_ret_raw)

TRUE_DELTA = np.array([0.5, 0.3, 0.2])
TRUE_RHO = np.array([0.3, 0.25, 0.35, 0.2, 0.15])
TRUE_BETA = np.array([1.0, 0.8, 1.2, 0.9, 0.7])
TRUE_SIGMA_ETA = np.array([0.01, 0.012, 0.015, 0.011, 0.009])

W_star = TRUE_DELTA[0]*W_vol + TRUE_DELTA[1]*W_tail + TRUE_DELTA[2]*W_ret
Lambda = np.diag(TRUE_RHO)
A = np.eye(N) - Lambda @ W_star
A_inv = np.linalg.inv(A)

# Generate factor
factor = np.zeros(T)
fv = 0.01
for t in range(1, T):
    fv = 0.001 + 0.85*fv + 0.1*factor[t-1]**2
    factor[t] = 0.02*factor[t-1] + np.sqrt(max(fv, 1e-8))*np.random.randn()

# Generate returns
eta = np.column_stack([TRUE_SIGMA_ETA[i]*np.random.standard_t(5, T) for i in range(N)])
R = np.zeros((T, N))
for t in range(T):
    R[t] = A_inv @ (TRUE_BETA * factor[t] + eta[t])

# PCA factor
Rd = R - R.mean(axis=0)
_, _, Vt = np.linalg.svd(Rd, full_matrices=False)
F = (Rd @ Vt[0]).reshape(-1, 1)
F = F / F.std()

print("="*70)
print("ORACLE TEST: MLE with TRUE networks (no Granger)")
print("="*70)
print(f"  True delta = {TRUE_DELTA}")
print(f"  True rho   = {TRUE_RHO}")
print()
print(f"{'Win':>5} | {'d_vol':>6} {'d_tail':>6} {'d_ret':>6} | "
      f"{'rho0':>6} {'rho1':>6} {'rho2':>6} {'rho3':>6} {'rho4':>6} | {'loglik':>9} | st")
print("-"*90)

W_list = [W_vol, W_tail, W_ret]

for win in [100, 200, 300, 500, 750, 1000]:
    R_win = R[T-win:]
    F_win = F[T-win:]
    t0 = time.time()
    res = estimate_single_window(
        R_win, F_win, W_list,
        rho_init_values=[-0.5, -0.1, 0.0, 0.1, 0.3, 0.5],
        maxiter=5000, ftol=1e-10
    )
    dt = time.time() - t0
    d = res['delta_hat']
    rho = res['lambda_hat']
    st = "OK" if res['success'] else "FAIL"
    print(f"{win:5d} | {d[0]:6.3f} {d[1]:6.3f} {d[2]:6.3f} | "
          f"{rho[0]:6.3f} {rho[1]:6.3f} {rho[2]:6.3f} {rho[3]:6.3f} {rho[4]:6.3f} | "
          f"{res['loglik']:9.1f} | {st} ({dt:.0f}s)", flush=True)

print(f"\nTrue: | {TRUE_DELTA[0]:6.3f} {TRUE_DELTA[1]:6.3f} {TRUE_DELTA[2]:6.3f} | "
      f"{TRUE_RHO[0]:6.3f} {TRUE_RHO[1]:6.3f} {TRUE_RHO[2]:6.3f} {TRUE_RHO[3]:6.3f} {TRUE_RHO[4]:6.3f} |")
