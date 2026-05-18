"""
Measure and visualize SMPL-X bodies produced by SMPLest-X inference.py.

Each inference .npz (demo/results/<name>/smplx/<frame>_<id>.npz) contains:
  betas       (10,)     — shape parameters (gender-neutral, T-pose)
  vertices    (10475,3) — posed, camera-space mesh (NOT suitable for measurement)

Usage:
  # single file
  python measure_smplestx.py path/to/000001_0.npz

  # whole directory  (averages repeated subjects across frames)
  python measure_smplestx.py demo/results/test/smplx/

  # specify gender if known
  python measure_smplestx.py demo/results/test/smplx/ --gender FEMALE
"""

import argparse
import http.server
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS, PROTECH_LABELS

GROUND_TRUTH = {
    "height":             181.5,
    "neck circumference":  41.0,
    "chest circumference": 112.0,
    "waist circumference":  98.0,
    "hip circumference":   104.0,
    "apex adjustment":      28.0,
    "front length":        117.0,
    "skirt length":         62.0,
}


def measure_npz(npz_path, gender: str = "NEUTRAL", posed: bool = False) -> MeasureBody:
    data = np.load(npz_path)
    betas = torch.tensor(data["betas"], dtype=torch.float32).unsqueeze(0)  # (1, 10)

    body_pose = None
    global_orient = None
    if posed:
        body_pose = torch.tensor(data["body_pose"], dtype=torch.float32).unsqueeze(0)
        global_orient = torch.zeros(1, 3, dtype=torch.float32)

    measurer = MeasureBody("smplx")
    measurer.from_body_model(gender=gender, shape=betas,
                             body_pose=body_pose,
                             global_orient=global_orient)
    measurer.measure(measurer.all_possible_measurements)
    measurer.label_measurements(PROTECH_LABELS)
    return measurer


def summarise(results) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        row = {"file": name}
        row.update({f"{label} ({m.labels2names[label]})": round(v, 1)
                    for label, v in m.labeled_measurements.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help=".npz file or directory of .npz files")
    parser.add_argument("--gender", default="NEUTRAL",
                        choices=["NEUTRAL", "MALE", "FEMALE"])
    parser.add_argument("--no-viz", action="store_true",
                        help="skip interactive 3-D visualisation")
    parser.add_argument("--posed", action="store_true",
                        help="reconstruct body in actual pose using body_pose from .npz")
    args = parser.parse_args()

    target = Path(args.path)
    npz_files = sorted(target.glob("*.npz")) if target.is_dir() else [target]

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found at {target}")

    results = {}
    for npz in npz_files:
        print(f"  measuring {npz.name} …")
        results[npz.stem] = measure_npz(npz, gender=args.gender, posed=args.posed)

    df = summarise(results)
    for file_name, m in results.items():
        print(f"\n=== {file_name} ===")
        for label, value in m.labeled_measurements.items():
            name = m.labels2names[label]
            print(f"  {label}  {name:<35} {value:.1f} cm")
        print(f"\n  --- errors vs ground truth ---")
        for m_name, gt_val in GROUND_TRUTH.items():
            if m_name not in m.measurements:
                continue
            err = m.measurements[m_name] - gt_val
            pct = err / gt_val * 100
            sign = "+" if err >= 0 else ""
            print(f"  {m_name:<35} {sign}{err:.1f} cm  ({sign}{pct:.1f}%)")

    if len(results) > 1:
        print("\n=== Mean across files ===")
        for col in df.columns:
            label, name = col.split(" ", 1)
            print(f"  {label}  {name:<35} {df[col].mean():.1f} cm")

        print("\n=== MAE vs ground truth ===")
        for m_name, gt_val in GROUND_TRUTH.items():
            values = [m.measurements[m_name] for m in results.values()
                      if m_name in m.measurements]
            if not values:
                continue
            mae = sum(abs(v - gt_val) for v in values) / len(values)
            pct = mae / gt_val * 100
            print(f"  {m_name:<35} {mae:.1f} cm  ({pct:.1f}%)")

    if not args.no_viz:
        first_name, first_m = next(iter(results.items()))
        print(f"\nVisualising {first_name} …")
        fig = first_m.visualize(title=first_name, show=False)

        # write figure to a temp HTML file and serve it so the
        # visualisation stays live until the user quits the script
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fig.write_html(tmp.name)
        tmp.close()

        serve_dir = os.path.dirname(tmp.name)
        html_file = os.path.basename(tmp.name)

        handler = http.server.SimpleHTTPRequestHandler
        httpd = http.server.HTTPServer(("0.0.0.0", 0), handler)
        port = httpd.server_address[1]

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        url_local = f"http://localhost:{port}/{html_file}"
        url_network = f"http://{local_ip}:{port}/{html_file}"
        print(f"\n  Local:   {url_local}")
        print(f"  Network: {url_network}")
        os.chdir(serve_dir)

        try:
            input("\nPress Enter (or Ctrl+C) to exit …\n")
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
