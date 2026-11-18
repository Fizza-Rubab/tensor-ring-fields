import math, torch, torch.nn.functional as Fn
def render(F, res=256, az=35, el=20, dev='cuda:0', steps=400):
    F = F.to(dev).float()[None, None]
    a, e = math.radians(az), math.radians(el)
    fwd = torch.tensor([-math.sin(a)*math.cos(e), -math.sin(e), -math.cos(a)*math.cos(e)], device=dev)
    up0 = torch.tensor([0., 1., 0.], device=dev); right = torch.linalg.cross(fwd, up0); right /= right.norm(); up = torch.linalg.cross(right, fwd)
    s = torch.linspace(-1.05, 1.05, res, device=dev); V, U = torch.meshgrid(-s, s, indexing='ij')
    o = U[..., None]*right + V[..., None]*up - 1.8*fwd
    t = torch.linspace(0, 3.6, steps, device=dev)
    P = o[:, :, None] + t[None, None, :, None]*fwd
    def f(P):
        g = P[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        return Fn.grid_sample(F, g, align_corners=False, padding_mode='border').reshape(P.shape[:-1])
    vals = f(P); inside = vals < 0
    hit = inside.any(-1); first = inside.float().argmax(-1)
    ph = o + t[first][..., None]*fwd
    h = 2/F.shape[-1]
    nrm = torch.stack([f(ph + h*torch.eye(3, device=dev)[i]) - f(ph - h*torch.eye(3, device=dev)[i]) for i in range(3)], -1)
    nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-8)
    L = -fwd*0.5 + up*0.6 + right*0.4; L = L/L.norm()
    shade = (0.25 + 0.75*(nrm*L).sum(-1).clamp(min=0))
    img = torch.where(hit, shade, torch.ones_like(shade))
    return img.cpu().numpy()
