import sys, os, time, numpy as np, igl, trimesh, torch, torch.nn.functional as Fn
from scipy import ndimage
from skimage.measure import marching_cubes
from multiprocessing import Pool

lst, out, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
NF, NC = int(os.environ.get("NF", 512)), 128
vf, vc = 2 / NF, 2 / NC


def name_of(p):
    parts = p.split("/")
    return (
        f"{parts[-4]}_{parts[-3]}"
        if p.endswith("model_normalized.obj")
        else "scan_" + os.path.basename(p)[:-4]
    )


def work(p):
    nm = name_of(p)
    fo = f"{out}/{nm}.npy"
    if os.path.exists(fo):
        return nm, "exists"
    try:
        t0 = time.time()
        torch.set_num_threads(1)
        m = trimesh.load(p, force="mesh", process=False)
        V = np.asarray(m.vertices, np.float64)
        F = np.asarray(m.faces, np.int64)
        c = (V.max(0) + V.min(0)) / 2
        V = (V - c) / np.abs(V - c).max() * 0.9
        tc = (np.arange(NC) + 0.5) / NC * 2 - 1
        Pc = np.stack(np.meshgrid(tc, tc, tc, indexing="ij"), -1).reshape(-1, 3)
        Uc, _, _ = igl.signed_distance(Pc, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)
        U = Fn.interpolate(
            torch.from_numpy(np.abs(Uc).reshape(NC, NC, NC).astype(np.float32))[None, None],
            size=(NF,) * 3,
            mode="trilinear",
            align_corners=False,
        )[0, 0].numpy()
        band = U < 2.5 * vc
        idx = np.argwhere(band)
        d, _, _ = igl.signed_distance(
            (idx + 0.5) / NF * 2 - 1, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED
        )
        U[band] = np.abs(d)
        lab, _ = ndimage.label(U >= 1.0 * vf)
        border = np.unique(
            np.concatenate(
                [
                    lab[0].ravel(),
                    lab[-1].ravel(),
                    lab[:, 0].ravel(),
                    lab[:, -1].ravel(),
                    lab[:, :, 0].ravel(),
                    lab[:, :, -1].ravel(),
                ]
            )
        )
        S = np.where(np.isin(lab, border[border > 0]), U, -U) - 0.75 * vf
        np.save(fo, S.astype(np.float16))
        v, fc, _, _ = marching_cubes(S, 0.0)
        v = v / NF * 2 - 1 + 1 / NF
        pts = trimesh.sample.sample_surface(trimesh.Trimesh(v, fc, process=False), 500000, seed=0)[0].astype(
            np.float32
        )
        np.save(f"{out}/{nm}_surf.npy", pts)
        return nm, f"inside {(S < 0).mean():.4f}, {len(F):,} faces, {time.time()-t0:.0f}s"
    except Exception as e:
        return nm, f"FAILED {e}"


paths = [l.strip() for l in open(lst) if l.strip()]
with Pool(nproc) as pool:
    for nm, msg in pool.imap_unordered(work, paths):
        print(nm, msg, flush=True)
