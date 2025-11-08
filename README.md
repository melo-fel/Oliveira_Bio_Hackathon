# Challenge 3 — Cell Lineage Trees & Division Analysis

Visualize cell lineage trees from tracking data, compute lineage statistics, and animate cell divisions.

<img width="1785" height="1219" alt="lineage_top5" src="https://github.com/user-attachments/assets/64dc339a-6a4e-4c95-bc9f-08a28cda7fb3" />

## What this repo does
1. **Loads a CSV** produced by a tracker (columns like Track_id, Parent_id, Start_frame, End_frame, Morphology_class, …).
2. **Builds lineage trees** by following Parent_id → Track_id up to the **root** (Parent_id == 0 or missing).
3. **Computes per-lineage stats**:
   - Nodes (tree size)
   - Divisions (parents that have **≥ 2 children** inside the same lineage)
   - Framespan = max(End_frame) - min(Start_frame) + 1
   - Divisions_per_frame = Divisions / Framespan
   - morph_* counts (distribution of morphology classes)
4. **Plots the top-K lineages** (largest by node count; ties by divisions) with **color-coded morphology**.
5. **Bonus animation**: shows the lineage evolving over frames with **live connections** between alive nodes.

## Install

*Python 3.9 and up is recommended*
pip install pandas numpy matplotlib networkx pillow

## Run in terminal

python lineage_challenge3.py --csv /path/to/your.csv --topk 5 --outdir ./outputs

