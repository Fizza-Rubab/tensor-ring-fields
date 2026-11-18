# Projective Tensor-Ring Fields

Code for the report *Projective Tensor-Ring Fields: Decoder-Free Plane Representations for 3D Shapes*.

A shape is stored as `n` feature planes whose pixels hold `4×4` matrices. The signed distance at a point is the trace of the product of the matrices fetched from all planes (a tensor ring). The planes are either the three axis planes (triplane) or `n` oblique planes along spherical-design directions.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is required for fitting.

## Data

Reference shapes are 512³ signed distance grids built from meshes (ShapeNet models and scans):

```bash
python build_references.py meshes.txt data/hq 16
```

`meshes.txt` lists one mesh path per line. Each shape produces `<name>.npy` (SDF) and `<name>_surf.npy` (500k surface points). The scripts read from `data/hq`, or from `$HQ_DIR` if set. The 100 evaluation shapes are drawn from this pool with a fixed seed.

## Usage

```bash
python ptr.py fit ring 6 64 cuda:0
python benchmark.py ring 6 64 cuda:0
python cost.py ring 6 64 cuda:0
python tables.py
```

`ptr.py fit <ring|prod> <planes> <budget> <gpu>` fits shapes with `n` planes at the memory of a `3×budget²` triplane. `benchmark.py` takes the same arguments (plus `mlpshared` for the shared-decoder baseline, trained first with `TRAINDEC=1`) and evaluates at 512³. `run_all.sh` runs every experiment.

## Files

| File | Purpose |
|---|---|
| `ptr.py` | Ring and product readouts on `n` planes, fitting, loop orders, truncation, DMRG-style sweeps |
| `benchmark.py` | Equal-memory benchmark with held-out error, IoU, Chamfer and F-score at 512³ |
| `placement_learned.py` | Axis+diagonal and learned six-plane layouts |
| `cost.py` | Arithmetic and decode time per readout |
| `tables.py`, `placement_table.py` | Aggregate results into tables |
| `fig_benchmark.py`, `fig_oblique.py`, `render.py` | Error-coloured figures |
| `build_references.py` | Watertight 512³ reference SDFs from meshes |

Useful environment variables: `SHARD=k/K` splits the 100 shapes across jobs, `NSUB` limits the number of shapes, `ITERS` sets the fitting steps (default 10000), `R` sets the matrix size, `ROT=1` rotates shapes randomly.
