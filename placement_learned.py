import sys, os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
mode, rep = sys.argv[1], sys.argv[2]
if mode == 'learn': layout, N3, dev = 'learn6', int(sys.argv[3]), sys.argv[4]
else: layout, N3, dev = sys.argv[3], int(sys.argv[4]), sys.argv[5]
D = os.environ.get('HQ_DIR', 'data/hq'); OUT = 'results/place100'; os.makedirs(OUT, exist_ok=True)
NF, vf, PTS, R, CH, TAU, ITERS = 512, 2 / 512, 32768, 4, 16, 8 * 2 / 512, int(os.environ.get('ITERS', 10000))
K = 3 if layout == 'axis3' else 6; NP = N3 if K == 3 else int(round(N3 / math.sqrt(2)))
allnames = sorted(f[:-4] for f in os.listdir(D) if f.endswith('.npy') and not f.endswith('_surf.npy'))
perm = np.random.RandomState(2).permutation(len(allnames))
EVAL = sorted(allnames[i] for i in perm[:100]); LEARN = sorted(allnames[i] for i in perm[100:120])
E3 = torch.eye(3); dg = lambda a, b: (a + b) / math.sqrt(2)
AXIS6 = torch.stack([torch.stack(f) for f in [(E3[0], E3[1]), (E3[1], E3[2]), (E3[2], E3[0]),
                                              (dg(E3[0], E3[1]), E3[2]), (dg(E3[1], E3[2]), E3[0]), (dg(E3[2], E3[0]), E3[1])]])
def ortho(E):
    a = E[:, 0] / E[:, 0].norm(dim=1, keepdim=True); b = E[:, 1] - (E[:, 1] * a).sum(1, keepdim=True) * a
    return torch.stack([a, b / b.norm(dim=1, keepdim=True)], 1)
def samp(P, u, v): return Fn.grid_sample(P, torch.stack([v, u], -1)[:, None], mode='bilinear', align_corners=False, padding_mode='border')[:, :, 0].permute(0, 2, 1)
def field(P, E, p):
    S = P.shape[0]; E = ortho(E); feats = []
    for k in range(K):
        a, b = E[k, 0], E[k, 1]
        feats.append(samp(P[:, k], (p @ a) / a.abs().sum(), (p @ b) / b.abs().sum()))
    if rep == 'prod':
        out = feats[0]
        for f in feats[1:]: out = out * f
        return out.sum(-1)
    M = [f.view(S, -1, R, R) for f in feats]; T = M[0]
    for m in M[1:-1]: T = T @ m
    return (T * M[-1].transpose(-1, -2)).sum((-1, -2))
init_scale = {('ring', 3): 0.6, ('ring', 6): 0.75, ('prod', 3): 0.4, ('prod', 6): 0.55}[(rep, K)]
def run(names, E, learn_E):
    S = len(names)
    G = torch.stack([torch.from_numpy(np.load(f'{D}/{n}.npy')) for n in names]).to(dev)[:, None].half()
    pools = torch.stack([(lambda bi: bi[torch.randint(len(bi), (400000,), device=dev)])(torch.nonzero(G[s, 0].abs() < 1.5 * vf)) for s in range(S)])
    ar = torch.arange(S, device=dev)[:, None]
    sdf_at = lambda p: Fn.grid_sample(G, p[..., [2, 1, 0]][:, :, None, None].half(), mode='bilinear', align_corners=False, padding_mode='border')[:, 0, :, 0, 0].float()
    def near(k, sig=(1.0, 4.0, 16.0)):
        vi = pools[ar, torch.randint(pools.shape[1], (S, k), device=dev)]
        sg = torch.tensor(sig, device=dev)[torch.randint(len(sig), (S, k), device=dev)][..., None] * vf
        return ((vi.float() + 0.5) / NF * 2 - 1 + torch.randn(S, k, 3, device=dev) * sg).clamp(-1, 1)
    torch.manual_seed(0)
    P0 = torch.randn(S, K, CH, NP, NP, device=dev) * 0.1
    P0 += (init_scale * torch.eye(R, device=dev).flatten())[None, None, :, None, None] if rep == 'ring' else init_scale
    P = nn.Parameter(P0); groups = [{'params': [P], 'lr': 1e-2}]
    if learn_E: E = nn.Parameter(E.clone()); groups.append({'params': [E], 'lr': 1e-3})
    opt = torch.optim.Adam(groups); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS); pts = PTS if S <= 10 else PTS // 2
    for it in range(ITERS):
        nu = pts // 4; p = torch.cat([torch.rand(S, nu, 3, device=dev) * 2 - 1, near(pts - nu)], 1)
        loss = (field(P, E, p) - sdf_at(p).clamp(-TAU, TAU)).abs().mean() / TAU
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0: print(f'  it {it} loss {loss.item():.4f}', flush=True)
    with torch.no_grad():
        err, err4 = torch.zeros(S, device=dev), torch.zeros(S, device=dev)
        for _ in range(20):
            q = near(10000, (1.0,)); err += (field(P, E, q) - sdf_at(q).clamp(-TAU, TAU)).abs().mean(1) / vf / 20
            q = near(10000, (4.0,)); err4 += (field(P, E, q) - sdf_at(q).clamp(-TAU, TAU)).abs().mean(1) / vf / 20
        if os.environ.get('SAVEP') and not learn_E:
            os.makedirs(f'{OUT}/planes', exist_ok=True)
            for s in range(S): torch.save({'P': P[s].detach().cpu(), 'E': ortho(E).cpu()}, f'{OUT}/planes/{rep}_{layout}_{N3}_{names[s]}.pt')
        if SAVEF and not learn_E:
            os.makedirs(f'{OUT}/fields', exist_ok=True); NR = 384; t = (torch.arange(NR, device=dev) + 0.5) / NR * 2 - 1; Xg, Yg = torch.meshgrid(t, t, indexing='ij')
            for s in range(S):
                Fv = torch.stack([field(P[s:s+1], E, torch.stack([torch.full_like(Xg, t[xi]), Xg, Yg], -1).reshape(1, -1, 3)).view(NR, NR) for xi in range(NR)])
                np.save(f'{OUT}/fields/{rep}_{layout}_{N3}_{names[s]}.npy', Fv.cpu().numpy().astype(np.float16))
    return E.detach(), err.cpu().numpy(), err4.cpu().numpy()
if mode == 'learn':
    E, err, _ = run(LEARN, AXIS6.to(dev), True); E = ortho(E).cpu()
    torch.save(E, f'{OUT}/frames_{rep}_{N3}.pt')
    for k in range(6):
        n = torch.linalg.cross(E[k, 0], E[k, 1]); print(f'plane {k}: normal {n.numpy().round(3)}  angle to init normal {math.degrees(math.acos(min(1, abs(float(n @ torch.linalg.cross(AXIS6[k, 0], AXIS6[k, 1])))))):.1f} deg')
    print(f'learn-set err@1vox {err.mean():.4f}'); sys.exit()
E = {'axis3': AXIS6[:3], 'axis6': AXIS6}.get(layout)
if layout == 'learn6': E = torch.load(f'{OUT}/frames_{rep}_{N3}.pt')
if os.environ.get('NAMES'): EVAL = [n for n in EVAL if any(n.startswith(x) for x in os.environ['NAMES'].split(','))]
SAVEF = int(os.environ.get('SAVEF', 0)); E = E.to(dev); res = []
for i in range(0, len(EVAL), 10):
    nb = EVAL[i:i+10]; _, err, err4 = run(nb, E, False)
    res += [{'name': n, 'rep': rep, 'layout': layout, 'N3': N3, 'NP': NP, 'K': K, 'params': K * CH * NP * NP,
             'heldout_vox': float(err[s]), 'heldout4_vox': float(err4[s])} for s, n in enumerate(nb)]
    print(f'{rep} {layout} {N3}: {i+len(nb)}/100  running mean {np.mean([r["heldout_vox"] for r in res]):.4f}', flush=True)
    torch.cuda.empty_cache()
json.dump(res, open(f'{OUT}/{"wins_" if os.environ.get("NAMES") else ""}{rep}_{layout}_{N3}.json', 'w'))
