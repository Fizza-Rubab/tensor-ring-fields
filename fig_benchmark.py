import sys, os, glob, math, numpy as np, torch, torch.nn.functional as Fn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
dev, OUTP = sys.argv[1], sys.argv[2]; sys.path.insert(0, os.getcwd()); from render import render
D = os.environ.get('HQ_DIR', 'data/hq'); vf = 2 / 512; FD = 'results/fig/fields'
SH = [('03467517_5c570a10', 'guitar', (20, 60)), ('04530566_30e45246', 'boat', (35, 25)), ('03636649_6bb8020f', 'lamp', (35, 20)), ('scan_alligator', 'alligator (scan)', (35, 35))]
M = [('prod_n3_64', 'Triplane product'), ('ring_n3_64', 'Triplane ring'), ('prod_n6_64', 'Oblique product'), ('ring_n6_64', 'Oblique ring'), ('mlpshared_n3_64', 'Triplane + shared MLP\n(learned decoder)')]
def gt_field(name, sc, Rm, NR=256):
    G = torch.from_numpy(np.load(f'{D}/{name}.npy')).to(dev)[None, None].float(); t = (torch.arange(NR, device=dev) + 0.5) / NR * 2 - 1
    X, Y, Z = torch.meshgrid(t, t, t, indexing='ij'); p = torch.stack([X, Y, Z], -1).reshape(-1, 3); x = (p @ Rm.to(dev)) / sc
    out = torch.cat([Fn.grid_sample(G, x[i:i + 2 ** 22, [2, 1, 0]][None, :, None, None], align_corners=False, padding_mode='border')[0, 0, :, 0, 0] for i in range(0, len(x), 2 ** 22)])
    return (out * sc).view(NR, NR, NR).cpu()
def ball(F):
    NR = F.shape[0]; r = ((torch.arange(NR) + 0.5) / NR * 2 - 1) ** 2; rad = (r[:, None, None] + r[None, :, None] + r[None, None]).sqrt()
    return torch.where(rad > 0.99, torch.full_like(F, 0.03), F)
def render_err(F, Ev, res=360, az=35, el=20, steps=500):
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
    return shade.cpu().numpy(), f(Ev, ph).cpu().numpy(), hit.cpu().numpy()
cmap = plt.get_cmap('inferno'); EMAX = 1.0
def colour(sh, er, hit): rgb = cmap(np.clip(er / EMAX, 0, 1))[..., :3] * sh[..., None]; rgb[~hit] = 1; return rgb
rows = []
for key, label, (az, el) in SH:
    fs = {m: torch.load(glob.glob(f'{FD}/{m}_{key}*.pt')[0]) for m, _ in M}; a = fs[M[0][0]]
    name = [os.path.basename(f)[:-4] for f in glob.glob(f'{D}/{key}*.npy') if not f.endswith('_surf.npy')][0]
    G = gt_field(name, a['sc'], a['R']); ims = [render(ball(G), res=360, az=az, el=el, dev=dev)]; errs = []
    for m, _ in M:
        F = fs[m]['field'].float(); sh, er, hit = render_err(ball(F), (F - G).abs() / (a['sc'] * vf), az=az, el=el); ims.append(colour(sh, er, hit)); errs.append(fs[m]['err'])
    rows.append((label, ims, errs)); print(label, ' '.join(f'{e:.3f}' for e in errs), flush=True)
fig, ax = plt.subplots(len(rows), 1 + len(M), figsize=(2.3 * (1 + len(M)), 2.35 * len(rows) + 0.2), squeeze=False)
for i, (label, ims, errs) in enumerate(rows):
    m_ = np.stack([(im if im.ndim == 2 else im.min(-1)) < 0.999 for im in ims]).any(0); ys, xs = np.where(m_); y0, y1, x0, x1 = max(0, ys.min() - 6), ys.max() + 6, max(0, xs.min() - 6), xs.max() + 6
    best = int(np.argmin(errs[:4]))
    for j, im in enumerate(ims):
        ax[i, j].imshow(im[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1) if im.ndim == 2 else ax[i, j].imshow(im[y0:y1, x0:x1]); ax[i, j].axis('off')
        if j == 0: ax[i, j].set_title(f'reference\n{label}', fontsize=8)
        else: ax[i, j].set_title(f'{M[j - 1][1]}\n{errs[j - 1]:.3f} vox', fontsize=7.5, fontweight='bold' if j - 1 == best else 'normal')
plt.tight_layout(rect=(0, 0.04, 1, 1))
x_sep = (ax[0, 4].get_position().x1 + ax[0, 5].get_position().x0) / 2
fig.add_artist(matplotlib.lines.Line2D([x_sep, x_sep], [0.06, 0.99], color='0.4', lw=1.0, ls='--'))
fig.text((ax[0, 1].get_position().x0 + ax[0, 4].get_position().x1) / 2, 0.995, 'decoder-free (no learned weights)', ha='center', va='top', fontsize=8, style='italic')
cax = fig.add_axes([0.3, 0.018, 0.4, 0.012]); cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, EMAX), cmap=cmap), cax=cax, orientation='horizontal')
cb.set_label('local distance error (voxels of the $512^3$ reference)', fontsize=8); cb.ax.tick_params(labelsize=7)
plt.savefig(OUTP, dpi=170); print('saved', OUTP)
