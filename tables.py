import json, glob, os, re, collections, numpy as np
from scipy.stats import wilcoxon

OUTD = "results/tables"
os.makedirs(OUTD, exist_ok=True)
R = {}
for f in [x for x in glob.glob("results/v2/*.jsonl") if not x.endswith("cost.jsonl")]:
    tag = re.sub(r"_sh\d+of\d+$", "", os.path.basename(f)[:-6])
    R.setdefault(tag, {}).update(
        {r["name"]: r for r in (json.loads(l) for l in open(f) if l.startswith("{"))}
    )


def common(*t):
    s = None
    for x in t:
        s = set(R.get(x, {})) if s is None else s & set(R.get(x, {}))
    return sorted(s or [])


def mean(tag, k, nm):
    v = np.array([R[tag][x][k] for x in nm])
    return float(np.nanmean(v))


def better(a, b, k, nm):
    x = np.array([R[a][i][k] for i in nm])
    y = np.array([R[b][i][k] for i in nm])
    w = (x > y) if k in ("iou", "f1") else (x < y)
    return w.mean() * 100, wilcoxon(x, y).pvalue


M = [
    ("prod_n3", "Triplane product", "no"),
    ("ring_n3", "Triplane ring", "no"),
    ("prod_n6", "Oblique product (6 planes)", "no"),
    ("ring_n6", "Oblique ring (6 planes)", "no"),
    ("mlpshared_n3", "Triplane + shared MLP decoder (learned)", "yes"),
]
K = [("heldout", "min"), ("iou", "max"), ("chamfer", "min"), ("f1", "max")]
lines = []
Nmin = 1000
FREE = [m for m in M if m[2] == "no"]
LEARN = [m for m in M if m[2] == "yes"]
for N3 in (64, 128):
    nm = common(*[f"{m}_{N3}" for m, _, _ in M])
    Nmin = min(Nmin, len(nm))
    vals = {m: [mean(f"{m}_{N3}", k, nm) for k, _ in K] for m, _, _ in M}
    bestfree = [(min if d == "min" else max)(vals[m][j] for m, _, _ in FREE) for j, (_, d) in enumerate(K)]
    fmt = lambda k, v: f"{v:.3f}" if k in ("heldout", "chamfer") else f"{v:.4f}"
    lines.append(
        f"\\midrule\\multicolumn{{5}}{{l}}{{\\emph{{budget $3\\times{N3}^2$ ({len(nm)} shapes)}}}}\\\\"
    )
    for m, lab, _ in FREE:
        lines.append(
            lab
            + " & "
            + " & ".join(
                (
                    f"\\textbf{{{fmt(k, vals[m][j])}}}"
                    if abs(vals[m][j] - bestfree[j]) < 1e-12
                    else fmt(k, vals[m][j])
                )
                for j, (k, _) in enumerate(K)
            )
            + " \\\\"
        )
    lines.append("\\cmidrule(lr){1-5}")
    for m, lab, _ in LEARN:
        lines.append(lab + " & " + " & ".join(fmt(k, vals[m][j]) for j, (k, _) in enumerate(K)) + " \\\\")
open(f"{OUTD}/main.tex", "w").write("\n".join(lines) + "\n")
open(f"{OUTD}/main_n.tex", "w").write(str(Nmin))
W = []
for N3 in (64, 128):
    for a, b, lab in [
        ("ring_n3", "prod_n3", "Triplane ring vs.\\ triplane product"),
        ("ring_n6", "prod_n6", "Oblique ring vs.\\ oblique product"),
        ("ring_n6", "ring_n3", "Oblique ring vs.\\ triplane ring"),
        ("prod_n6", "prod_n3", "Oblique product vs.\\ triplane product"),
        ("mlpshared_n3", "ring_n6", "Shared MLP vs.\\ oblique ring"),
    ]:
        nm = common(f"{a}_{N3}", f"{b}_{N3}")
        h, ph = better(f"{a}_{N3}", f"{b}_{N3}", "heldout", nm)
        c, pc = better(f"{a}_{N3}", f"{b}_{N3}", "chamfer", nm)
        pf = lambda p: (
            "$<10^{-15}$"
            if p < 1e-15
            else (f"{p:.2f}" if p >= 0.01 else f"$10^{{{int(np.floor(np.log10(p)))}}}$")
        )
        W.append(f"$3\\times{N3}^2$ & {lab} & {len(nm)} & {h:.0f}\\% & {pf(ph)} & {c:.0f}\\% & {pf(pc)} \\\\")
    W.append("\\midrule")
open(f"{OUTD}/wins.tex", "w").write("\n".join(W[:-1]) + "\n")
rt = {r: "ring_n6_64" + ("" if r == 4 else f"_R{r}") for r in (1, 2, 3, 4, 6)}
nm = common(*[t for t in rt.values() if t in R])
vals = {r: mean(t, "heldout", nm) for r, t in rt.items() if t in R}
b = min(vals.values())
row1 = " & ".join(str(r) for r in vals)
row2 = " & ".join(str(r * r) for r in vals)
row3 = " & ".join(f"{v:.3f}" for v in vals.values())
row4 = " & ".join(f"{(v / vals[4] - 1) * 100:+.0f}\\%" if r != 4 else "--" for r, v in vals.items())
open(f"{OUTD}/rank.tex", "w").write(
    f"Matrix size $r$ & {row1} \\\\\nNumbers per pixel ($r^2$, memory) & {row2} \\\\\n\\midrule\nHeld-out error (vox) & {row3} \\\\\nChange vs.\\ $r=4$ & {row4} \\\\\n"
)
open(f"{OUTD}/rank_n.tex", "w").write(str(len(nm)))
C = [json.loads(l) for l in open("results/v2/cost.jsonl") if l.startswith("{")]
LAB = {
    ("mlp", 3): "Triplane + MLP decoder",
    ("prod", 3): "Triplane product",
    ("ring", 3): "Triplane ring",
    ("prod", 6): "Oblique product, 6 planes",
    ("ring", 6): "Oblique ring, 6 planes",
    ("prod", 15): "Oblique product, 15 planes",
    ("ring", 15): "Oblique ring, 15 planes",
}
cl = []
for key, lab in LAB.items():
    a = next(c for c in C if (c["rep"], c["n"]) == key and c["N3"] == 64)
    b = next(c for c in C if (c["rep"], c["n"]) == key and c["N3"] == 128)
    mac = f"{a['mac']:,}".replace(",", "{,}")
    dec = f"{a['decoder']:,}".replace(",", "{,}") if a["decoder"] else "--"
    cl.append(
        f"{lab} & {dec} & {key[1]} & {mac} & {a['ms_1M']:.1f} & {b['ms_1M']:.1f} & {b['ms_fetch']:.1f} \\\\"
    )
open(f"{OUTD}/cost.tex", "w").write("\n".join(cl) + "\n")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sty = {
    "mlpshared_n3": ("#7f7f7f", "o", "-"),
    "prod_n3": ("#eb6834", "s", "--"),
    "ring_n3": ("#2a78d6", "s", "-"),
    "prod_n6": ("#eb6834", "D", ":"),
    "ring_n6": ("#1a4f9c", "D", "-"),
}
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
for j, (k, lab) in enumerate(
    [("heldout", "held-out distance error (voxels)"), ("chamfer", "Chamfer distance (voxels)")]
):
    for m, lab_, _ in M:
        c, mk, ls = sty[m]
        ys = []
        for N3 in (64, 128):
            nm = common(*[f"{q}_{N3}" for q, _, _ in M])
            ys.append(mean(f"{m}_{N3}", k, nm))
        ax[j].plot([64, 128], ys, color=c, marker=mk, ls=ls, label=lab_)
    ax[j].set_xticks([64, 128])
    ax[j].set_xticklabels(["$3\\times64^2$", "$3\\times128^2$"])
    ax[j].set_ylabel(lab, fontsize=8)
    ax[j].set_xlabel("memory budget", fontsize=8)
    ax[j].grid(alpha=0.3)
    ax[j].tick_params(labelsize=7)
ax[0].legend(fontsize=6.5, frameon=False)
plt.tight_layout()
plt.savefig("results/tables/budget.pdf")
print("tables written, N main", Nmin, "N rank", len(nm))
