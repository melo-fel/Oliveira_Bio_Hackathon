"""
Cell lineage visualization + stats

1) Loads a CSV with columns like:
   Track_id, Parent_id, Start_frame, End_frame, Morphology_class, Area, Length, Width, Centroid_x, Centroid_y

2) Groups rows into lineages (trees) by walking Parent_id up to a root (Parent_id == 0 or parent missing).

3) Computes per-lineage stats:
   - Nodes (size)
   - Divisions (count parents with >=2 children within the lineage)
   - Framespan = max(End_frame) - min(Start_frame) + 1
   - Divisions_per_frame = Divisions / Framespan
   - Morphology distribution (columns morph_*)

4) Plots the top-K largest lineages as trees with color-coded morphology.

5) BONUS: Animates the largest lineage across frames and saves a GIF.

-----

Felipe Garcia Cruz
"""
from __future__ import annotations
import os
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.animation import FuncAnimation, PillowWriter

# Load & Normalization

REQUIRED_COLUMNS = ["Track_id", "Parent_id", "Start_frame", "End_frame", "Morphology_class", "Area", "Length", "Width", "Centroid_x", "Centroid_y"]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    import re

    def clean(s: str) -> str:
        s = str(s).replace("\ufeff", "").replace("\xa0", " ")
        s = s.strip().lower()
        s = re.sub(r"[\s\-]+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        return s

    df = df.rename(columns={c: clean(c) for c in df.columns})

    alias = {
        "track_id": "Track_id", "trackid": "Track_id", "id": "Track_id",
        "parent_id": "Parent_id", "parentid": "Parent_id", "parent": "Parent_id",
        "start_frame": "Start_frame", "startframe": "Start_frame", "start": "Start_frame",
        "end_frame": "End_frame", "endframe": "End_frame", "end": "End_frame",
        "morphology_class": "Morphology_class", "morphology": "Morphology_class",
        "area": "Area", "length": "Length", "width": "Width",
        "centroid_x": "Centroid_x", "centroidy": "Centroid_y", "centroid_y": "Centroid_y",
    }
    df = df.rename(columns=alias)

    required = ["Track_id", "Parent_id", "Start_frame", "End_frame", "Morphology_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}. "
                         f"Seen: {list(df.columns)}")

    for col in ["Track_id", "Parent_id", "Start_frame", "End_frame"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["Area", "Length", "Width", "Centroid_x", "Centroid_y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Morphology_class"] = df["Morphology_class"].astype(str).str.strip()
    return df

def load_lineage_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # strips BOM if present
    return _normalize_cols(df)

# Lineage group & stats

def _find_root(track_id: int, parent_map: Dict[int, int]) -> int:
    seen = set()
    cur = track_id
    while True:
        if cur in seen:
            return cur
        seen.add(cur)
        p = parent_map.get(cur, 0)
        if p == 0 or p not in parent_map:
            return cur
        cur = p

def compute_roots(df: pd.DataFrame) -> Dict[int, int]:
    parent_map = dict(zip(df["Track_id"].astype(int), df["Parent_id"].fillna(0).astype(int)))
    roots: Dict[int, int] = {}
    for tid in parent_map:
        roots[tid] = _find_root(tid, parent_map)
    return roots

def build_child(df: pd.DataFrame) -> Dict[int, List[int]]:
    child: Dict[int, List[int]] = {}
    for _, r in df.iterrows():
        tid = int(r["Track_id"])
        pid = int(r["Parent_id"]) if not pd.isna(r["Parent_id"]) else 0
        if pid != 0:
            child.setdefault(pid, []).append(tid)
    return child

def lineage_group(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    roots = compute_roots(df)
    df = df.copy()
    df["Root_id"] = df["Track_id"].map(roots)
    return {rid: g.copy() for rid, g in df.groupby("Root_id")}

def _framespan(g: pd.DataFrame) -> int:
    return int(g["End_frame"].max() - g["Start_frame"].min() + 1)

def compute_lineage_stats(df: pd.DataFrame) -> pd.DataFrame:
    groups = lineage_group(df)
    childmap = build_child(df)

    rows = []
    for rid, g in groups.items():
        g_ids = set(g["Track_id"].astype(int).tolist())

        divide = 0
        for pid, kids in childmap.items():
            for pid in g_ids:
                k_in = [k for k in kids if k in g_ids]
                if len(k_in) >= 2:
                    divide += 1
        
        lifespan_frames = _framespan(g)
        morph_counts = g["Morphology_class"].str.lower().value_counts().to_dict()
        rows.append({
            "Root_id": int(rid),
            "Nodes": int(len(g)),
            "Divisions": int(divide),
            "Framespan": int(lifespan_frames),
            "Divisions_per_frame": divide / max(1, lifespan_frames),
            **{f"morph_{k}": int(v) for k, v in morph_counts.items()}
        })
    
    stats = pd.DataFrame(rows).sort_values(["Nodes", "Divisions"], ascending = [False, False]).reset_index(drop = True)
    return stats

# Visuals

DEFAULT_COLORS = {
    "elongated": "#5B8FF9",
    "healthy":   "#52C41A",
    "divided":   "#722ED1",
    "curved":    "#FA8C16",
}

def hierarchy_pos(G: nx.DiGraph, root: int, width = 1.0, vert_gap = 1.4, vert_loc = 0.0, xcenter = 0.5, pos = None):
    if pos is None:
        pos = {}
    
    pos[root] = (xcenter, vert_loc)
    child = list(G.successors(root))
    if not child:
        return pos
    
    dx = width / len(child)
    nextx = xcenter - width / 2 - dx / 2
    for ch in child:
        nextx += dx
        pos = hierarchy_pos(G, ch, width = dx, vert_gap = vert_gap, vert_loc = vert_loc - vert_gap, xcenter = nextx, pos = pos)
    
    return pos

def draw_lineage_tree(gdf: pd.DataFrame, title: str = "", color_map: Dict[str, str] | None = None, ax = None):
    if color_map is None:
        color_map = DEFAULT_COLORS
    
    G = nx.DiGraph()

    for _, r in gdf.iterrows():
        tid = int(r["Track_id"])
        G.add_node(
            tid,
            morph=str(r["Morphology_class"]).lower(),
            start=int(r["Start_frame"]),
            end=int(r["End_frame"]),
        )
    
    tid_to_pid = {int(r["Track_id"]): int(r["Parent_id"]) for _, r in gdf.iterrows()}
    for tid, pid in tid_to_pid.items():
        if pid != 0 and pid in G and tid in G:
            G.add_edge(pid, tid)
    
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    root = roots[0] if roots else list(G.nodes)[0]

    pos = hierarchy_pos(G, root)
    node_colors = [color_map.get(G.nodes[n]["morph"], "#999999") for n in G.nodes]

    if ax is None:
        fig, ax = plt.subplots(figsize = (4, 4))
    
    nx.draw(G, pos, with_labels = False, node_color = node_colors, node_size = 320, ax = ax, arrows = False)
    for n, (x, y) in pos.items():
        ax.text(x, y, str(n), fontsize = 8, ha = "center", va = "center", color = "white")
    
    ax.set_title(title, fontsize = 10)
    ax.axis("off")

    return G, pos

def plot_top_k_lineage(df: pd.DataFrame, k: int = 5, outfile: str | None = None):
    stats = compute_lineage_stats(df)
    top = stats.head(k)
    groups = lineage_group(df)

    k = len(top)
    cols = min(3, k if k else 1)
    rows = int(np.ceil(max(1, k) / cols))

    fig, axes = plt.subplots(rows, cols, figsize = (4 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (_, r) in enumerate(top.iterrows()):
        rid = int(r["Root_id"])
        gdf = groups[rid].sort_values("Start_frame")
        title = f"Lineage {i+1} (Root {rid})\nDivisions: {r['Divisions']} ({r['Divisions_per_frame']:.3f}/frame)"
        draw_lineage_tree(gdf, title = title, ax = axes[i])
    
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=lbl.capitalize()) for lbl, c in DEFAULT_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight", dpi=150)
    return stats, fig

# BONUS: Animation

def animate_lineage(df: pd.DataFrame, root_id: int, outfile: str = "lineage_anim.gif", fps: int = 5):
    groups = lineage_group(df)
    if root_id not in groups:
        raise ValueError(f"Root {root_id} not found.")
    gdf = groups[root_id].sort_values("Start_frame")

    # Build graph
    G = nx.DiGraph()
    for _, r in gdf.iterrows():
        tid = int(r["Track_id"])
        G.add_node(tid,
                   morph=str(r["Morphology_class"]).lower(),
                   start=int(r["Start_frame"]),
                   end=int(r["End_frame"]))
    for _, r in gdf.iterrows():
        tid = int(r["Track_id"]); pid = int(r["Parent_id"]) if not pd.isna(r["Parent_id"]) else 0
        if pid != 0 and pid in G and tid in G:
            G.add_edge(pid, tid)

    # Root + layout
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    root = roots[0] if roots else list(G.nodes)[0]
    pos = hierarchy_pos(G, root)
    all_edges = list(G.edges())

    # Time range
    t0 = int(gdf["Start_frame"].min())
    t1 = int(gdf["End_frame"].max())
    frames = list(range(t0, t1 + 1))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")

    def alive_nodes(t):
        return [n for n in G.nodes
                if int(G.nodes[n]["start"]) <= t <= int(G.nodes[n]["end"])]

    def update(t):
        ax.clear(); ax.axis("off")

        # 1) Background skeleton (entire tree)
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, ax=ax,
                               edge_color="#BBBBBB", width=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes), ax=ax,
                               node_color="#DDDDDD", node_size=180)
        nx.draw_networkx_labels(G, pos,
                                labels={n: str(n) for n in G.nodes},
                                font_size=6, font_color="#666666")

        # 2) Alive overlay: nodes + connections (bold)
        alive = set(alive_nodes(t))
        alive_edges = [(u, v) for (u, v) in all_edges if u in alive and v in alive]

        nx.draw_networkx_edges(G, pos, edgelist=alive_edges, ax=ax,
                               edge_color="black", width=2.0)
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(alive), ax=ax, node_size=360,
            node_color=[DEFAULT_COLORS.get(G.nodes[n]["morph"], "#999999") for n in alive]
        )
        nx.draw_networkx_labels(G, pos,
                                labels={n: str(n) for n in alive},
                                font_size=8, font_color="white")

        ax.set_title(f"Lineage {root_id}  |  Frame {t}", fontsize=11)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    anim.save(outfile, writer=writer)
    plt.close(fig)
    return outfile

def animate_topk(df, stats, out_dir, k=5, fps=5):
    top = stats.head(k)
    for _, r in top.iterrows():
        rid = int(r["Root_id"])
        outfile = os.path.join(out_dir, f"lineage_{rid}.gif")
        animate_lineage(df, rid, outfile=outfile, fps=fps)

# Runner & CLI

def run_all(csv_path: str, topk: int = 5, out_dir: str = ".",
           make_gif: bool = True) -> Dict[str, str]:
    df = load_lineage_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    fig_path = os.path.join(out_dir, "lineage_top5.png")
    stats, _ = plot_top_k_lineage(df, k=topk, outfile=fig_path)

    stats_path = os.path.join(out_dir, "lineage_stats.csv")
    stats.to_csv(stats_path, index=False)
    animate_topk(df, stats, out_dir, k=topk, fps=5)


    gif_path = ""
    if make_gif and not stats.empty:
        best_root = int(stats.iloc[0]["Root_id"])
        gif_path = os.path.join(out_dir, "lineage_anim.gif")
        animate_lineage(df, best_root, outfile=gif_path, fps=5)

    return {"fig_path": fig_path, "stats_path": stats_path, "gif_path": gif_path}

def _build_argparser():
    p = argparse.ArgumentParser(description="Visualize cell lineage trees and compute stats.")
    p.add_argument("--csv", required=True, help="Path to CSV file with lineage rows.")
    p.add_argument("--topk", type=int, default=5, help="How many top lineages to draw.")
    p.add_argument("--outdir", default=".", help="Where to save outputs.")
    p.add_argument("--no-gif", action="store_true", help="Skip the bonus GIF animation.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    results = run_all(args.csv, topk=args.topk, out_dir=args.outdir, make_gif=not args.no_gif)
    print("Saved:")
    for k, v in results.items():
        if v:
            print(f" - {k}: {v}")