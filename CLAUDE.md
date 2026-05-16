# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project measures anthropometric dimensions (lengths, circumferences, and geodesic surface distances) from SMPL and SMPL-X parametric body models. It supports T-posed bodies and pose-aware reconstruction from SMPLest-X `.npz` inference files.

## Setup

Body model `.pkl` files are **not included** and must be downloaded separately:
- `SMPL_{MALE,FEMALE,NEUTRAL}.pkl` → `data/smpl/`
- `SMPLX_{MALE,FEMALE,NEUTRAL}.pkl` → `data/smplx/`

Install dependencies:
```bash
pip install -r docker/requirements.txt
conda install conda-forge::tvb-gdist   # exact geodesic distances (Kirsanov)
```

Or use Docker:
```bash
cd docker && sh build.sh && sh run.sh <CODE_PATH>
```

**Always use the `smpl` conda env:**
```bash
conda run -n smpl python ...
```

## Running

Demo measurement on a zero-shaped neutral SMPL body:
```bash
conda run -n smpl python measure.py --measure_neutral_smpl_with_mean_shape
conda run -n smpl python measure.py --measure_neutral_smplx_with_mean_shape
```

Measure SMPLest-X inference output (single file or directory):
```bash
# T-pose (betas only)
conda run -n smpl python measure_smplestx.py data/test/000001_0.npz

# Pose-aware (uses body_pose from .npz, zeros global_orient)
conda run -n smpl python measure_smplestx.py data/test/000001_0.npz --posed

# Whole directory — prints per-file, mean, and MAE vs ground truth
conda run -n smpl python measure_smplestx.py data/test/ --no-viz
```

Evaluate MAE between two measurement sets:
```bash
conda run -n smpl python evaluate.py
```

Visualizations:
```bash
conda run -n smpl python visualize.py --visualize_smpl_and_smplx_face_segmentation
conda run -n smpl python visualize.py --visualize_smpl_and_smplx_joints
conda run -n smpl python visualize.py --visualize_smpl_and_smplx_landmarks
```

## Architecture

**Entry point:** `MeasureBody(model_type)` in `measure.py` — a factory that returns either `MeasureSMPL` or `MeasureSMPLX`, both inheriting from `Measurer`.

**Measurement pipeline:**
1. Body is initialized via `from_body_model(gender, shape, body_pose=None, global_orient=None)` (uses `smplx` library) or `from_verts(verts)` (bypasses shape params — useful for image-fitted meshes).
2. `measure(measurement_names)` dispatches based on `MEASUREMENT_TYPES`:
   - `LENGTH` → Euclidean distance between two landmark vertex indices
   - `CIRCUMFERENCE` → mesh-plane cut + convex hull perimeter
   - `GEODESIC_LENGTH` → exact geodesic distance along mesh surface (Kirsanov via `gdist`)
3. **Lengths** are Euclidean distances between landmark vertex indices. A landmark can be a tuple of two indices (their average is used).
4. **Circumferences** cut the mesh with a plane defined by a landmark point (origin) and a joint-to-joint vector (normal) using `trimesh.intersections.mesh_plane`. Multi-part slices are disambiguated using face segmentation (`data/smpl[x]/smpl[x]_body_parts_2_faces.json`), then the convex hull perimeter is computed. Pass `approximate_circumferences=False` to use raw contour instead of convex hull.
5. **Geodesic lengths** use `gdist.compute_gdist` (Kirsanov's exact algorithm). Definitions are tuples of N vertex indices; the measurement sums N-1 consecutive segment distances, allowing waypoints to force anatomically correct surface paths (e.g. routing front length through the nipple to stay on the anterior surface).

**Key files:**
- `measure.py` — `MeasureBody` factory; `Measurer` base class with `measure_length`, `measure_circumference`, `measure_geodesic_length`; `set_shape()` accepts optional `body_pose` / `global_orient`
- `measure_smplestx.py` — CLI for SMPLest-X `.npz` files; `GROUND_TRUTH` dict for MAE reporting; `--posed` flag
- `measurement_definitions.py` — `MEASUREMENT_TYPES`, `SMPLMeasurementDefinitions`, `SMPLXMeasurementDefinitions` (each with `LENGTHS`, `CIRCUMFERENCES`, `GEODESIC_LENGTHS`, `CIRCUMFERENCE_TO_BODYPARTS`, `possible_measurements`); also `STANDARD_LABELS` and `PROTECH_LABELS` (label → measurement name mappings)
- `landmark_definitions.py` — `SMPL_LANDMARK_INDICES` and `SMPLX_LANDMARK_INDICES` (vertex index lookups by anatomical name)
- `joint_definitions.py` — joint name → index mappings and joint counts for each model
- `utils.py` — `filter_body_part_slices`, `convex_hull_from_3D_points`, `load_face_segmentation`, and `point_segmentation_to_face_segmentation`
- `visualize.py` — `Visualizer` class using Plotly; geodesic paths rendered as surface lines via Dijkstra on the mesh edge graph

## Measurement Labels

`PROTECH_LABELS` is used by `measure_smplestx.py` (garment industry labels):

| Label | Measurement |
|-------|-------------|
| A | height |
| B | neck circumference |
| D | chest circumference |
| E | waist circumference |
| F | hip circumference |
| G | front length |
| H | skirt length |
| I | apex adjustment |

## Adding a New Measurement

1. Add the name and type (`LENGTH`, `CIRCUMFERENCE`, or `GEODESIC_LENGTH`) to `MEASUREMENT_TYPES` in `measurement_definitions.py`.
2. Add its definition to the relevant dict in `SMPLXMeasurementDefinitions` (and/or `SMPLMeasurementDefinitions`):
   - `LENGTHS`: two landmark keys
   - `CIRCUMFERENCES`: `{"LANDMARKS": [...], "JOINTS": [joint1, joint2]}`
   - `GEODESIC_LENGTHS`: tuple of N vertex indices (waypoints route the surface path)
3. For circumferences: add an entry to `CIRCUMFERENCE_TO_BODYPARTS`.
4. Add any new landmark vertex indices to `landmark_definitions.py`.
5. To include in ground-truth MAE reporting, add to `GROUND_TRUTH` in `measure_smplestx.py`.

All measurements are returned in **cm**.

## Focus Areas

Current and near-term work is focused on:
- **Measurement refinement** — tuning landmark positions and geodesic waypoints for accuracy
- **Visualization refinement** — surface path rendering, landmark placement review
- **SMPLX results analysis** — comparing T-pose vs posed measurements, MAE against ground truth, per-subject breakdown
