import sys, os, glob, numpy as np, torch, torch.nn.functional as Fn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
dev, OUTP = sys.argv[1], sys.argv[2]; sys.path.insert(0, os.getcwd()); from render import render
D = os.environ.get('HQ_DIR', 'data/hq'); vf = 2 / 512
SH = [('03467517_5c570a10', 'guitar', (20, 60)), ('04530566_30e45246', 'boat', (35, 25)), ('04090263_2e0fd930', 'rifle', (35, 25)), ('04379243_14bbf802', 'table', (35, 25))]
def load(n, key): f = glob.glob(f'results/ptr/planes/fit_ring_n{n}_128_rot0_def_R4_e11_fig_names_{key}*.pt'); assert len(f) == 1, (n, key, f); return torch.load(f[0])
def gt_field(name, sc, Rm, NR=256):
    G = torch.from_numpy(np.load(f'{D}/{name}.npy')).to(dev)[None, None].float(); t = (torch.arange(NR, device=dev) + 0.5) / NR * 2 - 1
    X, Y, Z = torch.meshgrid(t, t, t, indexing='ij'); p = torch.stack([X, Y, Z], -1).reshape(-1, 3); x = (p @ Rm.to(dev)) / sc
    out = torch.cat([Fn.grid_sample(G, x[i:i + 2 ** 22, [2, 1, 0]][None, :, None, None], align_corners=False, padding_mode='border')[0, 0, :, 0, 0] for i in range(0, len(x), 2 ** 22)])
    return (out * sc).view(NR, NR, NR).cpu()
def ball(F):
    NR = F.shape[0]; r = ((torch.arange(NR) + 0.5) / NR * 2 - 1) ** 2; rad = (r[:, None, None] + r[None, :, None] + r[None, None]).sqrt()
    return torch.where(rad > 0.99, torch.full_like(F, 0.03), F)
import math
def render_err(F, Ev, res=420, az=35, el=20, steps=500):
    F = F.to(dev).float()[None, None]; Ev = Ev.to(dev).float()[None, None]
    a, e = math.radians(az), math.radians(el); fwd = torch.tensor([-math.sin(a) * math.cos(e), -math.sin(e), -math.cos(a) * math.cos(e)], device=dev)
    up0 = torch.tensor([0., 1., 0.], device=dev); right = torch.linalg.cross(fwd, up0); right /= right.norm(); up = torch.linalg.cross(right, fwd)
    s_ = torch.linspace(-1.05, 1.05, res, device=dev); V, U = torch.meshgrid(-s_, s_, indexing='ij'); o = U[..., None] * right + V[..., None] * up - 1.8 * fwd
    t = torch.linspace(0, 3.6, steps, device=dev)
    def f(V3, P): return Fn.grid_sample(V3, P[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3), align_corners=False, padding_mode='border').reshape(P.shape[:-1])
    inside = torch.stack([f(F, o + t[k] * fwd) < 0 for k in range(steps)], -1); hit = inside.any(-1); first = inside.float().argmax(-1)
    ph = o + t[first][..., None] * fwd; h = 2 / F.shape[-1]
    nrm = torch.stack([f(F, ph + h * torch.eye(3, device=dev)[i]) - f(F, ph - h * torch.eye(3, device=dev)[i]) for i in range(3)], -1); nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-8)
    L = -fwd * 0.5 + up * 0.6 + right * 0.4; L = L / L.norm(); shade = 0.35 + 0.65 * (nrm * L).sum(-1).clamp(min=0)
    err = f(Ev, ph); return shade.cpu().numpy(), err.cpu().numpy(), hit.cpu().numpy()
cmap = plt.get_cmap('inferno'); EMAX = 1.0
def colour(sh, er, hit):
    rgb = cmap(np.clip(er / EMAX, 0, 1))[..., :3] * sh[..., None]; rgb[~hit] = 1; return rgb
rows = []
for key, label, (az, el) in SH:
    a, b = load(3, key), load(15, key); name = [os.path.basename(f)[:-4] for f in glob.glob(f'{D}/{key}*.npy') if not f.endswith('_surf.npy')][0]
    G = gt_field(name, a['sc'], a['R']); F3, F15 = a['field'].float(), b['field'].float()
    ims = [render(ball(G), res=420, az=az, el=el, dev=dev)]; sv = a['sc'] * vf
    for F in (F3, F15):
        sh, er, hit = render_err(ball(F), (F - G).abs() / sv, az=az, el=el); ims.append(colour(sh, er, hit))
    rows.append((label, ims, a['err'], b['err'], (az, el), [None, a['E'].numpy(), b['E'].numpy()])); print(label, f"n3 {a['err']:.3f} n15 {b['err']:.3f}", flush=True)
fig, ax = plt.subplots(len(rows), 3, figsize=(9.6, 2.9 * len(rows)), squeeze=False)
def project(P, az, el, res=420):
    a_, e_ = math.radians(az), math.radians(el); fwd = np.array([-math.sin(a_) * math.cos(e_), -math.sin(e_), -math.cos(a_) * math.cos(e_)])
    right = np.cross(fwd, [0., 1., 0.]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return (P @ right + 1.05) / 2.1 * (res - 1), (1.05 - P @ up) / 2.1 * (res - 1)
PCOL3 = ['#1f9bd1', '#2ca25f', '#f0a30a']
def draw_planes(ax_, E, az, el, half=0.62):
    for k in range(len(E)):
        a_, b_ = E[k, 0], E[k, 1]; C = np.array([half * (sa * a_ + sb * b_) for sa, sb in [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]])
        x_, y_ = project(C, az, el); col = PCOL3[k] if len(E) == 3 else '#3fb6f2'
        ax_.plot(x_, y_, color=col, lw=1.3 if len(E) == 3 else 0.7, alpha=0.95 if len(E) == 3 else 0.6)
for i, (label, ims, e3, e15, (az, el), Es) in enumerate(rows):
    m = np.stack([(im if im.ndim == 2 else im.min(-1)) < 0.999 for im in ims]).any(0); ys, xs = np.where(m); y0, y1, x0, x1 = max(0, ys.min() - 8), ys.max() + 8, max(0, xs.min() - 8), xs.max() + 8
    for j, (im, t) in enumerate(zip(ims, [f'reference ({label})', f'triplane ring, 3 axis planes\nerror {e3:.3f} vox', f'projective ring, 15 oblique planes\nerror {e15:.3f} vox ({(e15 - e3) / e3 * 100:+.0f}%)'])):
        ax[i, j].imshow(im, cmap='gray', vmin=0, vmax=1) if im.ndim == 2 else ax[i, j].imshow(im)
        if Es[j] is not None: draw_planes(ax[i, j], Es[j], az, el)
        ax[i, j].set_xlim(x0, x1); ax[i, j].set_ylim(y1, y0); ax[i, j].set_title(t, fontsize=9); ax[i, j].axis('off')
plt.tight_layout(rect=(0, 0.035, 1, 1))
cax = fig.add_axes([0.3, 0.015, 0.4, 0.012]); cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, EMAX), cmap=cmap), cax=cax, orientation='horizontal')
cb.set_label('local distance error on the fitted surface (voxels of the $512^3$ reference)', fontsize=8); cb.ax.tick_params(labelsize=7)
plt.savefig(OUTP, dpi=170); print('saved', OUTP)
