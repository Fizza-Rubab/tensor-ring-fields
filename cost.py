import sys, os, json, time, numpy as np, torch
REP, n, N3, dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sys.argv = ['ptr.py', 'fit', 'prod' if REP == 'mlp' else REP, str(n), str(N3), dev]; import ptr as T
S, M = 1, 2 ** 20; P = torch.randn(S, n, 16, T.NP, T.NP, device=dev) * 0.3
if REP == 'mlp':
    W = [torch.randn(S, 16, 64, device=dev) * 0.3, torch.zeros(S, 64, device=dev), torch.randn(S, 64, 64, device=dev) * 0.2, torch.zeros(S, 64, device=dev), torch.randn(S, 64, 1, device=dev) * 0.2, torch.zeros(S, 1, device=dev)]
    def comb(F, mask=None):
        h = sum(F); h = torch.relu(torch.baddbmm(W[1][:, None], h, W[0])); h = torch.relu(torch.baddbmm(W[3][:, None], h, W[2])); return torch.baddbmm(W[5][:, None], h, W[4])[..., 0]
    T.combine = comb
p = torch.randn(S, M, 3, device=dev); p = p / p.norm(dim=-1, keepdim=True) * torch.rand(S, M, 1, device=dev) ** (1 / 3)
def timed(fn):
    for _ in range(5): fn()
    ts = []
    for _ in range(30): torch.cuda.synchronize(); t0 = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    return float(np.median(ts)) * 1e3
def ring_fast(P, p):
    F = T.feats(P, p); Ms = [f.reshape(-1, T.R, T.R) for f in F]; X = Ms[0]
    for m in Ms[1:-1]: X = torch.bmm(X, m)
    return (X * Ms[-1].transpose(1, 2)).sum((1, 2))
with torch.no_grad():
    t_all = timed(lambda: T.field(P, p)); t_fetch = timed(lambda: T.feats(P, p))
    if REP == 'ring':
        ref = T.field(P, p).flatten(); assert torch.allclose(ring_fast(P, p), ref, rtol=1e-3, atol=1e-3 * ref.abs().max().item())
        t_all = min(t_all, timed(lambda: ring_fast(P, p)))
mac = {'prod': 16 * n, 'ring': (n - 2) * T.R ** 3 + T.R ** 2, 'mlp': 16 * (n - 1) + 16 * 64 + 64 * 64 + 64}[REP]
print(json.dumps({'rep': REP, 'n': n, 'N3': N3, 'NP': T.NP, 'numbers': n * 16 * T.NP ** 2, 'decoder': 5313 if REP == 'mlp' else 0, 'mac': mac, 'ms_1M': t_all, 'ms_fetch': t_fetch}))
