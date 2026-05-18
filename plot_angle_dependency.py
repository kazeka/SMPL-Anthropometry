"""
For each 360-video directory under test/, measure all 72 frames and plot:
  - Top subplot:    10 beta coefficients vs view angle
  - Bottom subplot: per-measurement error (predicted - GT) in cm vs view angle

The 72 frames are assumed to span 0–355 ° in equal steps of 5 °.

Usage:
  conda run -n smpl python plot_angle_dependency.py
  conda run -n smpl python plot_angle_dependency.py --gender FEMALE
  conda run -n smpl python plot_angle_dependency.py test/sizing-rotate
"""

import argparse
import http.server
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from measure_smplestx import measure_npz, GROUND_TRUTH

N_FRAMES = 72
ANGLES = np.linspace(0, 360, N_FRAMES, endpoint=False)

BETA_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
ERROR_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f",
]


def process_video(video_dir: Path, gender: str) -> dict:
    npz_files = sorted(video_dir.glob("*.npz"))
    if len(npz_files) != N_FRAMES:
        print(f"  WARNING: expected {N_FRAMES} files, got {len(npz_files)}")

    betas_matrix = []   # (n_frames, 10)
    errors = {m: [] for m in GROUND_TRUTH}

    for i, npz in enumerate(npz_files):
        print(f"  [{i+1:2d}/{len(npz_files)}] {npz.name}", end="\r", flush=True)
        m = measure_npz(npz, gender=gender)
        data = np.load(npz)
        betas_matrix.append(data["betas"])
        for m_name, gt_val in GROUND_TRUTH.items():
            if m_name in m.measurements:
                errors[m_name].append(m.measurements[m_name] - gt_val)
            else:
                errors[m_name].append(float("nan"))

    print()
    angles = ANGLES[: len(npz_files)]
    return {"angles": angles, "betas": np.array(betas_matrix), "errors": errors}


def build_figure(video_name: str, data: dict) -> go.Figure:
    angles = data["angles"]
    betas = data["betas"]   # (n_frames, 10)
    errors = data["errors"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Beta coefficients", "Measurement error vs ground truth (cm)"),
        vertical_spacing=0.10,
        row_heights=[0.45, 0.55],
    )

    # --- betas ---
    for i in range(betas.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=angles, y=betas[:, i],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=BETA_COLORS[i], width=1.5),
                name=f"β{i}",
                legendgroup="betas",
                legendgrouptitle_text="Betas" if i == 0 else None,
            ),
            row=1, col=1,
        )

    # --- errors ---
    gt_names = list(GROUND_TRUTH.keys())
    for j, m_name in enumerate(gt_names):
        vals = errors[m_name]
        fig.add_trace(
            go.Scatter(
                x=angles, y=vals,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=ERROR_COLORS[j % len(ERROR_COLORS)], width=1.5),
                name=m_name,
                legendgroup="errors",
                legendgrouptitle_text="Errors" if j == 0 else None,
            ),
            row=2, col=1,
        )

    # zero-line for errors
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"Angle-of-view dependency — {video_name}", font_size=16),
        height=800,
        legend=dict(groupclick="toggleitem"),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="View angle (°)", row=2, col=1, dtick=30)
    fig.update_yaxes(title_text="Beta value", row=1, col=1)
    fig.update_yaxes(title_text="Error (cm)", row=2, col=1, zeroline=True)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="test",
                        help="root dir containing video sub-dirs, or a single video dir")
    parser.add_argument("--gender", default="NEUTRAL",
                        choices=["NEUTRAL", "MALE", "FEMALE"])
    parser.add_argument("--no-viz", action="store_true",
                        help="save HTML files but do not serve them")
    args = parser.parse_args()

    root = Path(args.path)
    if list(root.glob("*.npz")):
        video_dirs = [root]
    else:
        video_dirs = sorted(d for d in root.iterdir() if d.is_dir() and list(d.glob("*.npz")))

    if not video_dirs:
        raise FileNotFoundError(f"No video directories with .npz files found under {root}")

    html_files = []
    for vdir in video_dirs:
        print(f"\n=== {vdir.name} ===")
        data = process_video(vdir, args.gender)
        fig = build_figure(vdir.name, data)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", prefix=f"{vdir.name}_", delete=False
        )
        fig.write_html(tmp.name)
        tmp.close()
        html_files.append(tmp.name)
        print(f"  saved → {tmp.name}")

    if args.no_viz or not html_files:
        return

    # serve all HTML files from the same temp dir
    serve_dir = os.path.dirname(html_files[0])

    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("0.0.0.0", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print("\nServing plots:")
    for f in html_files:
        name = os.path.basename(f)
        print(f"  http://localhost:{port}/{name}")
        print(f"  http://{local_ip}:{port}/{name}")

    os.chdir(serve_dir)
    try:
        input("\nPress Enter (or Ctrl+C) to exit …\n")
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        for f in html_files:
            os.unlink(f)


if __name__ == "__main__":
    main()
