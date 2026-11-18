import sys, os, json, math, time, zlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
mode, rep, n, N3, dev = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ROT, ORDER = int(os.environ.get('ROT', 0)), os.environ.get('ORDER', 'def'); R = int(os.environ.get('R', 6 if mode == 'trunc' else 4))
D = os.environ.get('HQ_DIR', 'data/hq'); OUT = 'results/ptr'; os.makedirs(OUT, exist_ok=True)
NF, vf, PTS, TAU, ITERS = 512, 2 / 512, 16384, 8 * 2 / 512, int(os.environ.get('ITERS', 10000))
NP = int(round(N3 * math.sqrt(3 / n))); C = R * R if rep == 'ring' else 16; OBC = ORDER == 'obc'
allnames = sorted(f[:-4] for f in os.listdir(D) if f.endswith('.npy') and not f.endswith('_surf.npy'))
EVAL = sorted(allnames[i] for i in np.random.RandomState(2).permutation(len(allnames))[:100])
if os.environ.get('NSUB'): EVAL = EVAL[:int(os.environ['NSUB'])]
if os.environ.get('NAMES'): EVAL = [x for x in EVAL if any(x.startswith(p) for p in os.environ['NAMES'].split(','))]
if os.environ.get('SHARD'): k_, K_ = map(int, os.environ['SHARD'].split('/')); EVAL = EVAL[k_::K_]
phi = (1 + 5 ** .5) / 2
def normals(n):
    t = lambda *a: torch.tensor(a, dtype=torch.float64); cyc = lambda v: [v, v[[1, 2, 0]], v[[2, 0, 1]]]
    N = {3: lambda: torch.eye(3, dtype=torch.float64),
         4: lambda: t([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]),
         6: lambda: torch.stack(cyc(t(0, 1, phi)) + cyc(t(0, 1, -phi))),
         10: lambda: torch.stack([t(1, 1, 1), t(1, 1, -1), t(1, -1, 1), t(-1, 1, 1)] + cyc(t(0, 1 / phi, phi)) + cyc(t(0, 1 / phi, -phi))),
         15: lambda: torch.stack(list(torch.eye(3, dtype=torch.float64)) + sum([cyc(t(*s)) for s in [(1, phi, phi * phi), (-1, phi, phi * phi), (1, -phi, phi * phi), (1, phi, -phi * phi)]], []))}[n]()
    return N / N.norm(dim=1, keepdim=True)
def frames(N):
    E = []
    for nv in N:
        e = torch.eye(3, dtype=torch.float64)[nv.abs().argmin()]; a = torch.linalg.cross(nv, e); a = a / a.norm()
        E.append(torch.stack([a, torch.linalg.cross(nv, a)]))
    return torch.stack(E).float()
NRM = normals(n); E = frames(NRM).to(dev)
def loop_order():
    if ORDER in ('def', 'obc'): return list(range(n))
    if ORDER.startswith('rand'): return [int(i) for i in np.random.RandomState(int(ORDER[4:])).permutation(n)]
    G = (NRM @ NRM.T).abs().numpy(); o = [0]
    while len(o) < n:
        c = [j for j in range(n) if j not in o]; o.append(max(c, key=lambda j: G[o[-1], j]) if ORDER == 'nn' else min(c, key=lambda j: G[o[-1], j]))
    return o
ORD = loop_order()
def rotation(name):
    if not ROT: return np.eye(3)
    q, r = np.linalg.qr(np.random.RandomState(zlib.crc32(name.encode())).randn(3, 3)); q = q * np.sign(np.diag(r)); return q * np.linalg.det(q)
class Data:
    def __init__(s, names):
        S = len(names); s.S = S; s.G = torch.stack([torch.from_numpy(np.load(f'{D}/{nm}.npy')) for nm in names]).to(dev)[:, None].half()
        s.sc = torch.tensor([0.98 / np.linalg.norm(np.load(f'{D}/{nm}_surf.npy'), axis=1).max() for nm in names], device=dev).float()[:, None, None]
        s.Rm = torch.tensor(np.stack([rotation(nm) for nm in names]), device=dev).float()
        s.pools = torch.stack([(lambda bi: bi[torch.randint(len(bi), (400000,), device=dev)])(torch.nonzero(s.G[i, 0].abs() < 1.5 * vf)) for i in range(S)])
        s.ar = torch.arange(S, device=dev)[:, None]
    def sdf(s, p):
        x = (p @ s.Rm) / s.sc
        return (Fn.grid_sample(s.G, x[..., [2, 1, 0]][:, :, None, None].half(), mode='bilinear', align_corners=False, padding_mode='border')[:, 0, :, 0, 0].float() * s.sc[:, :, 0]).clamp(-TAU, TAU)
    def near(s, k, sig=(1.0, 4.0, 16.0)):
        vi = s.pools[s.ar, torch.randint(s.pools.shape[1], (s.S, k), device=dev)]
        sg = torch.tensor(sig, device=dev)[torch.randint(len(sig), (s.S, k), device=dev)][..., None] * vf
        x = (vi.float() + 0.5) / NF * 2 - 1 + torch.randn(s.S, k, 3, device=dev) * sg
        return (s.sc * x) @ s.Rm.transpose(1, 2)
    def ball(s, k): d = torch.randn(s.S, k, 3, device=dev); return d / d.norm(dim=-1, keepdim=True) * torch.rand(s.S, k, 1, device=dev) ** (1 / 3)
    def train_pts(s, m): nu = m // 4; return torch.cat([s.ball(nu), s.near(m - nu)], 1)
def samp(P, u, v): return Fn.grid_sample(P, torch.stack([v, u], -1)[:, None], mode='bilinear', align_corners=False, padding_mode='border')[:, :, 0].permute(0, 2, 1)
def feats(P, p): return [samp(P[:, k], p @ E[k, 0], p @ E[k, 1]) for k in ORD]
def combine(F, mask=None):
    if rep == 'prod':
        out = F[0]
        for f in F[1:]: out = out * f
        return out.sum(-1)
    S, M = F[0].shape[:2]; Ms = [f.view(S, M, R, R) for f in F]
    if mask is not None: Ms = [m * mask[:, j, None] for j, m in enumerate(Ms)]
    T = Ms[0]
    for m in Ms[1:]: T = T @ m
    return T[..., 0, 0] if OBC else T.diagonal(dim1=-2, dim2=-1).sum(-1)
def field(P, p, mask=None): return combine(feats(P, p), mask)
SEED = int(os.environ.get('SEED', 0))
def init(S):
    torch.manual_seed(SEED); P0 = torch.randn(S, n, C, NP, NP, device=dev) * float(os.environ.get('NOISE', 0.1))
    if rep == 'ring' and os.environ.get('INIT') == 'e11':
        P0[:, :, 0] += 0.8 ** (1 / n)
    elif rep == 'ring': P0 += ((0.8 if OBC else 0.8 / R) ** (1 / n) * torch.eye(R, device=dev).flatten())[None, None, :, None, None]
    elif os.environ.get('INIT') == 'e11': P0[:, :, 0] += 0.8 ** (1 / n)
    else: P0 += (0.8 / C) ** (1 / n)
    global OFF; torch.manual_seed(SEED); OFF = P0 - torch.randn(S, n, C, NP, NP, device=dev) * float(os.environ.get('NOISE', 0.1))
    return P0
@torch.no_grad()
def heldout(P, d, reps=20, mask=None):
    e1, e4 = torch.zeros(d.S, device=dev), torch.zeros(d.S, device=dev)
    for _ in range(reps):
        q = d.near(10000, (1.0,)); e1 += (field(P, q, mask) - d.sdf(q)).abs().mean(1) / vf / reps
        q = d.near(10000, (4.0,)); e4 += (field(P, q, mask) - d.sdf(q)).abs().mean(1) / vf / reps
    return e1, e4
REG_TV, REG_L2, OFF = float(os.environ.get('REG_TV', 0)), float(os.environ.get('REG_L2', 0)), None
def adam(P0, d, iters, lr=1e-2, mask=None, curve=None, it0=0, tt=0.0):
    P = nn.Parameter(P0.clone()); opt = torch.optim.Adam([P], lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        torch.cuda.synchronize(); t0 = time.time(); p = d.train_pts(PTS)
        loss = (field(P, p, mask) - d.sdf(p)).abs().mean() / TAU
        if REG_TV and mask is None: loss = loss + REG_TV * ((P[..., 1:, :] - P[..., :-1, :]).pow(2).mean() + (P[..., 1:] - P[..., :-1]).pow(2).mean())
        if REG_L2 and mask is None: loss = loss + REG_L2 * (P - OFF).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); torch.cuda.synchronize(); tt += time.time() - t0
        if curve is not None and ((it + 1) % 500 == 0 or it in (49, 99, 199)): curve.append((it0 + it + 1, tt, heldout(P.detach(), d, 4)[0].tolist()))
        if it % 2000 == 0: print(f'  it {it} loss {loss.item():.4f}', flush=True)
    return P.detach()
PREC = os.environ.get('ALS_PREC', 'block')
def site_blocks(u, v, Et, w, ch=16384):
    S, M = u.shape; iy = (((u + 1) * NP - 1) / 2).clamp(0, NP - 1); ix = (((v + 1) * NP - 1) / 2).clamp(0, NP - 1)
    y0, x0 = iy.floor(), ix.floor(); fy, fx = iy - y0, ix - x0; y0, x0 = y0.long(), x0.long(); y1, x1 = (y0 + 1).clamp_max(NP - 1), (x0 + 1).clamp_max(NP - 1)
    cs = [(y0, x0, (1 - fy) * (1 - fx)), (y0, x1, (1 - fy) * fx), (y1, x0, fy * (1 - fx)), (y1, x1, fy * fx)]
    B = torch.zeros(S, NP * NP, C * C, device=dev)
    for a in range(0, M, ch):
        o = (w[:, a:a + ch, None, None] * Et[:, a:a + ch, :, None] * Et[:, a:a + ch, None, :]).flatten(-2)
        for yy, xx, ww in cs: B.scatter_add_(1, (yy[:, a:a + ch] * NP + xx[:, a:a + ch])[..., None].expand(-1, -1, C * C), ww[:, a:a + ch, None] ** 2 * o)
    return B.view(S, NP * NP, C, C)
def als(P0, d, sweeps, curve=None):
    M, cg, lam_rel, delta = int(os.environ.get('ALS_M', 262144)), int(os.environ.get('ALS_CG', 25)), float(os.environ.get('ALS_LAM', 1e-3)), float(os.environ.get('ALS_DELTA', 0.1)); lam0 = lam_rel
    P = P0.clone(); S = d.S; tt = 0.0
    def env_t(Pre, Suf):
        if rep == 'prod': return Pre * Suf
        if OBC: return (Pre[..., 0, :, None] * Suf[..., None, :, 0]).flatten(-2)
        return (Suf @ Pre).transpose(-1, -2).flatten(-2)
    EXT = int(os.environ.get('ALS_EXTRAP', 0)); beta = torch.ones(S, 1, 1, 1, 1, device=dev)
    def tloss(Q, q, yq): return (field(Q, q) - yq).abs().mean(1)
    for sw in range(sweeps):
        torch.cuda.synchronize(); t0 = time.time(); Pold = P.clone()
        with torch.no_grad():
            p = d.train_pts(M); y = d.sdf(p); F = feats(P, p)
            if rep == 'ring': F = [f.view(S, M, R, R) for f in F]
            one = torch.ones(S, M, C, device=dev) if rep == 'prod' else torch.eye(R, device=dev).expand(S, M, R, R)
            Suf = [None] * (n + 1); Suf[n] = one
            for j in range(n - 1, -1, -1): Suf[j] = F[j] * Suf[j + 1] if rep == 'prod' else F[j] @ Suf[j + 1]
            Pre = one
        LE = os.environ.get('ALS_LAMEND'); lam_rel = lam0 * (float(LE) / lam0) ** (sw / max(1, sweeps - 1)) if LE else lam0
        for j in range(n):
            k = ORD[j]; u, v = p @ E[k, 0], p @ E[k, 1]
            with torch.no_grad():
                Et = env_t(Pre, Suf[j + 1]); fcur = (F[j].flatten(2) * Et).sum(-1)
                w = 1 / (fcur - y).abs().clamp_min(delta * TAU)
            def JT(r, Et_):
                a = torch.zeros(S, C, NP, NP, device=dev, requires_grad=True)
                with torch.enable_grad(): return torch.autograd.grad((samp(a, u, v) * Et_).sum(-1), a, grad_outputs=r)[0]
            J = lambda a: (samp(a, u, v) * Et).sum(-1)
            with torch.no_grad():
                if PREC == 'block':
                    B = site_blocks(u, v, Et, w); lam = lam_rel * B.diagonal(dim1=-2, dim2=-1).mean((1, 2))[:, None, None, None] + 1e-8
                    Binv = torch.linalg.inv(B + lam * torch.eye(C, device=dev)); lam = lam[:, :, :, 0, None]
                    Mi = lambda r: (Binv @ r.flatten(2).transpose(1, 2)[..., None])[..., 0].transpose(1, 2).reshape(r.shape)
                else:
                    dg = JT(w, Et * Et); lam = lam_rel * dg.mean((1, 2, 3), keepdim=True) + 1e-8; Mi = lambda r, D_=1 / (dg + lam): D_ * r
                Aop = lambda a: JT(w * J(a), Et) + lam * a
                x = P[:, k].clone(); b = JT(w * y, Et) + lam * x; r = b - Aop(x); z = Mi(r); q = z.clone(); rz = (r * z).sum((1, 2, 3), keepdim=True)
                for _ in range(cg):
                    Aq = Aop(q); al = rz / (q * Aq).sum((1, 2, 3), keepdim=True).clamp_min(1e-30); x = x + al * q; r = r - al * Aq
                    z = Mi(r); rz2 = (r * z).sum((1, 2, 3), keepdim=True); q = z + rz2 / rz.clamp_min(1e-30) * q; rz = rz2
                P[:, k] = x; Fj = samp(x, u, v)
                Fj = Fj if rep == 'prod' else Fj.view(S, M, R, R); Pre = Pre * Fj if rep == 'prod' else Pre @ Fj
        del F, Suf, Pre
        if EXT and sw > 0:
            with torch.no_grad():
                q = d.train_pts(65536); yq = d.sdf(q); Pe = P + beta * (P - Pold); ok = tloss(Pe, q, yq) < tloss(P, q, yq)
                P = torch.where(ok[:, None, None, None, None], Pe, P); beta = torch.where(ok[:, None, None, None, None], (beta * 1.5).clamp_max(8), (beta / 2).clamp_min(0.25))
        torch.cuda.synchronize(); tt += time.time() - t0
        e = heldout(P, d, 4)[0]
        if curve is not None: curve.append((sw + 1, tt, e.tolist()))
        print(f'  sweep {sw + 1}: heldout {e.mean():.4f}  {tt:.0f}s', flush=True)
    if os.environ.get('ALS_ADAM'): P = adam(P, d, int(os.environ['ALS_ADAM']), lr=float(os.environ.get('ALS_ADAM_LR', 3e-3)), curve=curve, it0=10 ** 6, tt=tt)
    return P
@torch.no_grad()
def bond_svd(P, d, M=65536):
    p = d.train_pts(M); S = d.S; F = [f.view(S, M, R, R) for f in feats(P, p)]; m = n // 2; out = []
    def prod(idx):
        T = F[idx[0] % n]
        for i in idx[1:]: T = T @ F[i % n]
        return T
    def sqrtm(G): ev, V = torch.linalg.eigh(G.double()); return (V * ev.clamp_min(1e-12).sqrt()[:, None]) @ V.transpose(-1, -2)
    for k in range(n):
        L = prod(list(range(k - m + 1, k + 1))); Rr = prod(list(range(k + 1, k + 1 + n - m)))
        X = sqrtm(torch.einsum('smji,smjl->sil', L, L) / M); Y = sqrtm(torch.einsum('smij,smlj->sil', Rr, Rr) / M)
        U, Sv, Vh = torch.linalg.svd(X @ Y); out.append((X, Y, U, Sv, Vh.transpose(-1, -2)))
    return out
@torch.no_grad()
def truncate(P, B, ranks):
    S = P.shape[0]; Pq, Qt = [], []
    for k in range(n):
        X, Y, U, Sv, V = B[k]; msk = (torch.arange(R, device=dev)[None] < ranks[:, k, None]).double(); isq = msk / Sv.clamp_min(1e-30).sqrt()
        Pq.append((Y @ V) * isq[:, None, :]); Qt.append(isq[:, :, None] * (U.transpose(-1, -2) @ X))
    P2 = P.clone().view(S, n, R, R, NP, NP)
    for j in range(n):
        k = ORD[j]; A = P2[:, k].double()
        P2[:, k] = torch.einsum('sab,sbcxy,scd->sadxy', Qt[(j - 1) % n], A, Pq[j]).float()
    mask = torch.stack([(torch.arange(R, device=dev)[None, :, None] < ranks[:, (j - 1) % n, None, None]) & (torch.arange(R, device=dev)[None, None, :] < ranks[:, j, None, None]) for j in range(n)], 1).float()
    return P2.view(S, n, C, NP, NP), mask
if __name__ == '__main__':
    tag = f'{rep}_n{n}_{N3}_rot{ROT}_{ORDER}_R{R}' + ('_e11' if os.environ.get('INIT') == 'e11' else '') + (f'_sh{os.environ["SHARD"].replace("/", "of")}' if os.environ.get('SHARD') else '') + (f'_sub{os.environ["NSUB"]}' if os.environ.get('NSUB') else '') + os.environ.get('TAGX', '') + ('_names' if os.environ.get('NAMES') else '') + (f'_seed{SEED}' if SEED else '')
    print(f'{mode} {tag}: NP {NP}, C {C}, params/shape {n * C * NP * NP}, loop order {ORD}, {len(EVAL)} shapes', flush=True)
    res = []; BS = 10
    for i in range(0, len(EVAL), BS):
        nb = EVAL[i:i + BS]; d = Data(nb); P0 = init(len(nb)); cur = [] if (os.environ.get('CURVE') or mode == 'als') else None
        base = {'rep': rep, 'n': n, 'N3': N3, 'NP': NP, 'R': R, 'rot': ROT, 'order': ORDER, 'params': n * C * NP * NP}
        if mode in ('fit', 'als'):
            torch.cuda.synchronize(); t0 = time.time()
            P = adam(P0, d, ITERS, curve=cur) if mode == 'fit' else als(P0, d, int(os.environ.get('SWEEPS', 30)), curve=cur)
            wall = time.time() - t0; e1, e4 = heldout(P, d)
            if os.environ.get('PROBE'):
                for sg in [0.5, 1.0, 2.0]:
                    r_ = int(3 * sg + 1); g_ = torch.exp(-torch.arange(-r_, r_ + 1, device=dev).float() ** 2 / (2 * sg * sg)); g_ = g_ / g_.sum()
                    Q = P.reshape(-1, 1, NP, NP); Q = Fn.conv2d(Fn.pad(Q, (r_, r_, 0, 0), mode='replicate'), g_.view(1, 1, 1, -1)); Q = Fn.conv2d(Fn.pad(Q, (0, 0, r_, r_), mode='replicate'), g_.view(1, 1, -1, 1))
                    eb, _ = heldout(Q.view_as(P), d, 10); print(f'PROBE blur sigma {sg}: err {eb.mean():.4f} (unblurred {e1.mean():.4f}, x{(eb / e1).mean():.2f})', flush=True)
            if os.environ.get('SAVEP'):
                os.makedirs(f'{OUT}/planes', exist_ok=True); NR = 256; t = (torch.arange(NR, device=dev) + 0.5) / NR * 2 - 1; Xg, Yg = torch.meshgrid(t, t, indexing='ij')
                for s_, nm in enumerate(nb):
                    with torch.no_grad(): Fv = torch.stack([field(P[s_:s_ + 1], torch.stack([torch.full_like(Xg, t[xi]), Xg, Yg], -1).reshape(1, -1, 3)).view(NR, NR) for xi in range(NR)])
                    torch.save({'P': P[s_].cpu(), 'E': E.cpu(), 'ORD': ORD, 'sc': float(d.sc[s_]), 'R': d.Rm[s_].cpu(), 'field': Fv.half().cpu(), 'err': float(e1[s_])}, f'{OUT}/planes/{mode}_{tag}_{nm}.pt')
            for s, nm in enumerate(nb):
                r = dict(base, name=nm, heldout_vox=float(e1[s]), heldout4_vox=float(e4[s]), wall=wall / len(nb))
                if cur is not None: r['curve'] = [(c[0], c[1], c[2][s]) for c in cur]
                res.append(r)
        elif mode == 'trunc':
            P = adam(P0, d, ITERS); e1, _ = heldout(P, d); B = bond_svd(P, d); S = len(nb); en = torch.stack([b[3] for b in B], 1) ** 2
            sets = [('r', r, torch.full((S, n), r, device=dev)) for r in range(R - 1, 0, -1)]
            for eps in [0.3, 0.1, 0.03, 0.01]:
                tail = en.flip(-1).cumsum(-1).flip(-1) / en.sum(-1, keepdim=True)
                sets.append(('eps', eps, (tail > eps ** 2).sum(-1).clamp_min(1)))
            out = [dict(base, name=nm, full_vox=float(e1[s]), spectra=(en[s] / en[s].sum(-1, keepdim=True)).tolist(), sets=[]) for s, nm in enumerate(nb)]
            for kind, val, rk in sets:
                P2, mask = truncate(P, B, rk); a1, _ = heldout(P2, d, 10, mask); P3 = adam(P2, d, 1000, lr=3e-3, mask=mask); b1, _ = heldout(P3, d, 10, mask)
                mem = (NP * NP * torch.stack([rk[:, (j - 1) % n] * rk[:, j] for j in range(n)], 1).sum(1)).tolist()
                for s in range(S): out[s]['sets'].append({'kind': kind, 'val': val, 'ranks': rk[s].tolist(), 'params': mem[s], 'trunc_vox': float(a1[s]), 'ft_vox': float(b1[s])})
                print(f'  {kind}={val}: mean params {np.mean(mem):.0f}  trunc {a1.mean():.4f}  +ft {b1.mean():.4f}  (full {e1.mean():.4f})', flush=True)
            res += out
        print(f'{mode} {tag}: {i + len(nb)}/{len(EVAL)}  running mean {np.mean([r.get("heldout_vox", r.get("full_vox")) for r in res]):.4f}', flush=True)
        del d; torch.cuda.empty_cache()
    json.dump(res, open(f'{OUT}/{mode}_{tag}.json', 'w'))
