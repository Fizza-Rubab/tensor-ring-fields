import sys, os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

torch.set_num_threads(2)
REP, n, N3, dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.environ.setdefault("INIT", "e11")
os.environ.setdefault("ITERS", "10000")
MLP = REP in ("mlp", "mlpshared")
sys.argv = ["ptr.py", "fit", "prod" if MLP else REP, str(n), str(N3), dev]
import ptr as T
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
import trimesh

OUT = os.environ.get("OUT", "results/v2")
os.makedirs(OUT, exist_ok=True)
ITERS = T.ITERS
NF, vf, D = T.NF, T.vf, T.D
tag = (
    f"{REP}_n{n}_{N3}"
    + (f"_R{T.R}" if REP == "ring" and T.R != 4 else "")
    + ("_rot" if T.ROT else "")
    + os.environ.get("TAGX", "")
)
outp = (
    f"{OUT}/{tag}"
    + (f'_sh{os.environ["SHARD"].replace("/", "of")}' if os.environ.get("SHARD") else "")
    + ".jsonl"
)
try:
    os.environ.get("TRAINDEC") or os.close(os.open(outp + ".lock", os.O_CREAT | os.O_EXCL))
except FileExistsError:
    print("locked, skipping", outp, flush=True)
    sys.exit(0)
import glob as _g

done = set(
    json.loads(l)["name"]
    for f_ in _g.glob(f"{OUT}/{tag}.jsonl") + _g.glob(f"{OUT}/{tag}_sh*.jsonl")
    for l in open(f_)
)
TRAINDEC = REP == "mlpshared" and bool(os.environ.get("TRAINDEC"))
DEC = f"{OUT}/decoder_{N3}.pt"
if TRAINDEC:
    pool = [x for x in T.allnames if x not in T.EVAL]
    names = [pool[i] for i in np.random.RandomState(5).permutation(len(pool))[:10]]
else:
    names = [x for x in T.EVAL if x not in done]
W = None


def mlp_combine(F, mask=None):
    h = sum(F)
    S_ = h.shape[0]
    Wx = [w.expand(S_, *w.shape[1:]) for w in W]
    h = torch.relu(torch.baddbmm(Wx[1][:, None], h, Wx[0]))
    h = torch.relu(torch.baddbmm(Wx[3][:, None], h, Wx[2]))
    return torch.baddbmm(Wx[5][:, None], h, Wx[4])[..., 0]


def _unused(F):
    h = sum(F)
    h = torch.relu(torch.baddbmm(W[1][:, None], h, W[0]))
    h = torch.relu(torch.baddbmm(W[3][:, None], h, W[2]))
    return torch.baddbmm(W[5][:, None], h, W[4])[..., 0]


if MLP:
    T.combine = mlp_combine


def init(S):
    if not MLP:
        return T.init(S), []
    global W
    torch.manual_seed(0)
    if REP == "mlpshared" and not TRAINDEC:
        W = [w.to(dev) for w in torch.load(DEC)]
        return torch.randn(S, n, 16, T.NP, T.NP, device=dev) * 0.1, []
    if REP == "mlpshared":
        S = 1
    P0 = torch.randn(S, n, 16, T.NP, T.NP, device=dev) * 0.1
    lin = lambda i, o: [
        nn.Parameter(torch.randn(S, i, o, device=dev) * (2 / i) ** 0.5),
        nn.Parameter(torch.zeros(S, o, device=dev)),
    ]
    W = lin(16, 64)[:1] + [nn.Parameter(torch.zeros(S, 1, 64, device=dev)[:, 0])] + lin(64, 64) + lin(64, 1)
    with torch.no_grad():
        W[5].fill_(T.TAU)
    if REP == "mlpshared":
        P0 = torch.randn(len(names), n, 16, T.NP, T.NP, device=dev) * 0.1
    return P0, W


def fit(P0, extra, d):
    P = nn.Parameter(P0.clone())
    opt = torch.optim.Adam([P] + extra, 1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(ITERS):
        p = d.train_pts(T.PTS)
        loss = (T.field(P, p) - d.sdf(p)).abs().mean() / T.TAU
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return P.detach()


@torch.no_grad()
def grid_field(P1, sc, Rm):
    t = (torch.arange(NF, device=dev) + 0.5) / NF * 2 - 1
    X, Y = torch.meshgrid(t, t, indexing="ij")
    out = torch.empty(NF, NF, NF, dtype=torch.float16)
    for xi in range(NF):
        x = torch.stack([torch.full_like(X, t[xi]), X, Y], -1).reshape(1, -1, 3)
        p = (sc * x) @ Rm.T
        f = T.field(P1, p)[0]
        f = torch.where(p[0].norm(dim=-1) > 0.99, torch.full_like(f, T.TAU), f) / sc
        out[xi] = f.view(NF, NF).half().cpu()
    return out.float().numpy()


SH_ = " shared" if REP == "mlpshared" else ""
print(
    f"{tag}: NP {T.NP}, numbers/shape {n * 16 * T.NP * T.NP} (+{5313 if MLP else 0} decoder{SH_}), {len(names)} shapes left",
    flush=True,
)
BS = 10
for i in range(0, len(names), BS):
    nb = names[i : i + BS]
    d = T.Data(nb)
    P0, extra = init(len(nb))
    torch.cuda.synchronize()
    t0 = time.time()
    P = fit(P0, extra, d)
    torch.cuda.synchronize()
    tfit = (time.time() - t0) / len(nb)
    e1, _ = T.heldout(P, d)
    Wfull = W
    for s, nm in enumerate(nb):
        t1 = time.time()
        if REP == "mlp":
            W = [w[s : s + 1] for w in Wfull]
        if TRAINDEC:
            continue
        if os.environ.get("SAVEDIR"):
            os.makedirs(os.environ["SAVEDIR"], exist_ok=True)
            NR = 256
            t_ = (torch.arange(NR, device=dev) + 0.5) / NR * 2 - 1
            Xg, Yg = torch.meshgrid(t_, t_, indexing="ij")
            with torch.no_grad():
                Fv = torch.stack(
                    [
                        T.field(
                            P[s : s + 1],
                            torch.stack([torch.full_like(Xg, t_[xi]), Xg, Yg], -1).reshape(1, -1, 3),
                        ).view(NR, NR)
                        for xi in range(NR)
                    ]
                )
            torch.save(
                {"field": Fv.half().cpu(), "sc": float(d.sc[s]), "R": d.Rm[s].cpu(), "err": float(e1[s])},
                f"{os.environ['SAVEDIR']}/{tag}_{nm}.pt",
            )
        if os.environ.get("NOEVAL"):
            W = Wfull
            with open(outp, "a") as fo:
                fo.write(
                    json.dumps(
                        {
                            "name": nm,
                            "rep": REP,
                            "n": n,
                            "N3": N3,
                            "R": T.R,
                            "NP": T.NP,
                            "heldout": float(e1[s]),
                            "iou": float("nan"),
                            "chamfer": float("nan"),
                            "f1": float("nan"),
                            "f2": float("nan"),
                            "fit_s": tfit,
                            "eval_s": 0.0,
                        }
                    )
                    + "\n"
                )
            continue
        import fcntl

        while True:
            got = None
            for k_ in range(4):
                fh = open(f"{OUT}/.evalslot{k_}", "w")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    got = fh
                    break
                except BlockingIOError:
                    fh.close()
            if got:
                break
            time.sleep(2)
        Fp = grid_field(P[s : s + 1], float(d.sc[s]), d.Rm[s])
        W = Wfull
        Gt = d.G[s, 0].float().cpu().numpy()
        pin, gin = Fp < 0, Gt < 0
        iou = float((pin & gin).sum() / max((pin | gin).sum(), 1))
        ref = np.load(f"{D}/{nm}_surf.npy")[::2]
        if os.environ.get("NOEVAL"):
            cd, f1, f2 = float("nan"), float("nan"), float("nan")
        elif Fp.min() < 0 < Fp.max():
            v, fc, _, _ = marching_cubes(Fp, 0.0)
            v = v / NF * 2 - 1 + 1 / NF
            pred = trimesh.sample.sample_surface(trimesh.Trimesh(v, fc, process=False), 250000, seed=s)[0]
            dab = cKDTree(ref).query(pred, workers=2)[0]
            dba = cKDTree(pred).query(ref, workers=2)[0]
            cd = float((dab.mean() + dba.mean()) / 2 / vf)
            fsc = lambda th: float(
                2 * (dab < th).mean() * (dba < th).mean() / ((dab < th).mean() + (dba < th).mean() + 1e-9)
            )
            f1, f2 = fsc(vf), fsc(2 * vf)
        else:
            cd, f1, f2 = float("nan"), 0.0, 0.0
        del Fp, Gt, pin, gin
        fcntl.flock(got, fcntl.LOCK_UN)
        got.close()
        with open(outp, "a") as fo:
            fo.write(
                json.dumps(
                    {
                        "name": nm,
                        "rep": REP,
                        "n": n,
                        "N3": N3,
                        "R": T.R,
                        "NP": T.NP,
                        "heldout": float(e1[s]),
                        "iou": iou,
                        "chamfer": cd,
                        "f1": f1,
                        "f2": f2,
                        "fit_s": tfit,
                        "eval_s": time.time() - t1,
                    }
                )
                + "\n"
            )
    if TRAINDEC:
        torch.save([w.detach().cpu() for w in W], DEC)
        print("saved decoder", DEC, "train held-out", float(e1.mean()), flush=True)
        break
    print(
        f"{tag}: {i + len(nb)}/{len(names)}  fit {tfit:.1f}s/shape  eval {time.time() - t1:.1f}s/shape  heldout {e1.mean():.4f}",
        flush=True,
    )
    del d
    torch.cuda.empty_cache()
print("DONE", flush=True)
