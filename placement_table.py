import json, glob, re, collections, numpy as np
from scipy.stats import wilcoxon
d = collections.defaultdict(dict)
for f in glob.glob('results/ptr/fit_*.json'):
    t = re.sub(r'_sh\dof\d', '', f.split('fit_')[1][:-5])
    if any(x in t for x in ('names', 'sub', 'fig')): continue
    for r in json.load(open(f)): d[t][r['name']] = r['heldout_vox']
for f in glob.glob('results/place100/*_*_*.json'):
    b = f.split('/')[-1][:-5]
    if b.startswith('wins'): continue
    for r in json.load(open(f)): d['P_' + b][r['name']] = r['heldout_vox']
m = lambda t: np.mean(list(d[t].values())) if len(d.get(t, {})) >= 100 else None
def win(a, b):
    if len(d.get(a, {})) < 100 or len(d.get(b, {})) < 100: return ''
    k = sorted(set(d[a]) & set(d[b])); x = np.array([d[a][i] for i in k]); y = np.array([d[b][i] for i in k]); return f'{(x < y).mean() * 100:.0f}%'
f3 = lambda v: '-' if v is None else f'{v:.3f}'
for N3 in (128, 64):
    print(f'\n### budget 3x{N3}^2\n| layout | n | ring upright | prod upright | ring rotated | prod rotated | ring rot. penalty | ring better (upright) |\n|---|---|---|---|---|---|---|---|')
    rows = [('axis (triplane)', 3, 'def'), ('cube diagonals', 4, 'def'), ('axis + diagonals', 6, 'P_axis6'), ('learned frames', 6, 'P_learn6'), ('icosahedron', 6, 'def'), ('dodecahedron', 10, 'def'), ('icosidodecahedron', 15, 'def'),
            ('axis (triplane), E11', 3, 'e11'), ('icosahedron, E11', 6, 'e11'), ('icosidodecahedron, E11', 15, 'e11')]
    for lab, n, kind in rows:
        if kind.startswith('P_'):
            ru, pu = f'P_ring_{kind[2:]}_{N3}', f'P_prod_{kind[2:]}_{N3}'; rr = pr = None
        else:
            sfx = '_e11' if kind == 'e11' else ''; ru, pu = f'ring_n{n}_{N3}_rot0_def_R4{sfx}', f'prod_n{n}_{N3}_rot0_def_R4{sfx}'; rr, pr = f'ring_n{n}_{N3}_rot1_def_R4{sfx}', f'prod_n{n}_{N3}_rot1_def_R4{sfx}'
        a, b, c, e = m(ru), m(pu), m(rr) if rr else None, m(pr) if pr else None
        pen = '-' if (a is None or c is None) else f'{(c - a) / a * 100:+.1f}%'
        print(f'| {lab} | {n} | {f3(a)} | {f3(b)} | {f3(c)} | {f3(e)} | {pen} | {win(ru, pu)} |')
