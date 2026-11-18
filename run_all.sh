#!/bin/bash
set -e
G=${GPU:-cuda:0}

for N3 in 64 128; do
  TRAINDEC=1 python benchmark.py mlpshared 3 $N3 $G
  for cfg in "prod 3" "ring 3" "prod 6" "ring 6" "mlpshared 3"; do
    python benchmark.py $cfg $N3 $G
  done
done

for r in 1 2 3 6; do
  R=$r NOEVAL=1 NSUB=50 python benchmark.py ring 6 64 $G
done

for cfg in "mlp 3" "prod 3" "ring 3" "prod 6" "ring 6" "prod 15" "ring 15"; do
  for N3 in 64 128; do python cost.py $cfg $N3 $G >> results/v2/cost.jsonl; done
done

for rot in 0 1; do for N3 in 64 128; do for n in 3 4 6 10 15; do for rep in ring prod; do
  ROT=$rot python ptr.py fit $rep $n $N3 $G
  case $n in 3|6|15) INIT=e11 ROT=$rot python ptr.py fit $rep $n $N3 $G ;; esac
done; done; done; done

for rep in ring prod; do for N3 in 64 128; do
  python placement_learned.py learn $rep $N3 $G
  for layout in axis3 axis6 learn6; do python placement_learned.py fit $rep $layout $N3 $G; done
done; done

for n in 6 10; do for order in nn far rand0 rand1 obc; do
  ORDER=$order NSUB=30 python ptr.py fit ring $n 64 $G
done; done

NSUB=10 python ptr.py trunc ring 6 64 $G
NSUB=10 python ptr.py als ring 6 64 $G

python tables.py
python placement_table.py

for cfg in "prod 3" "ring 3" "prod 6" "ring 6"; do
  OUT=results/fig SAVEDIR=results/fig/fields NOEVAL=1 NAMES=03467517_5c570a10,04530566_30e45246,03636649_6bb8020f,scan_alligator python benchmark.py $cfg 64 $G
done
cp results/v2/decoder_64.pt results/fig/
OUT=results/fig SAVEDIR=results/fig/fields NOEVAL=1 NAMES=03467517_5c570a10,04530566_30e45246,03636649_6bb8020f,scan_alligator python benchmark.py mlpshared 3 64 $G
python fig_benchmark.py $G results/fig_benchmark.png

for n in 3 15; do
  INIT=e11 SAVEP=1 TAGX=_fig NAMES=03467517_5c570a10,04530566_30e45246,04090263_2e0fd930,04379243_14bbf802 python ptr.py fit ring $n 128 $G
done
python fig_oblique.py $G results/fig_oblique.png
