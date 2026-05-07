# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project measures anthropometric dimensions (lengths and circumferences) from SMPL and SMPL-X parametric body models. It operates on T-posed bodies only — for posed subjects, see the companion repo `pose-independent-anthropometry`.

## Setup

Body model `.pkl` files are **not included** and must be downloaded separately:
- `SMPL_{MALE,FEMALE,NEUTRAL}.pkl` → `data/smpl/`
- `SMPLX_{MALE,FEMALE,NEUTRAL}.pkl` → `data/smplx/`

Install dependencies:
```bash
pip install -r docker/requirements.txt
```

Or use Docker:
```bash
cd docker && sh build.sh && sh run.sh <CODE_PATH>
```

## Running

Demo measurement on a zero-shaped neutral SMPL body:
```bash
python measure.py --measure_neutral_smpl_with_mean_shape
python measure.py --measure_neutral_smplx_with_mean_shape
```

Evaluate MAE between two measurement sets:
```bash
python evaluate.py
```

Visualizations:
```bash
python visualize.py --visualize_smpl_and_smplx_face_segmentation
python visualize.py --visualize_smpl_and_smplx_joints
python visualize.py --visualize_smpl_and_smplx_landmarks
```

## Architecture

**Entry point:** `MeasureBody(model_type)` in `measure.py` — a factory that returns either `MeasureSMPL` or `MeasureSMPLX`, both inheriting from `Measurer`.

**Measurement pipeline:**
1. Body is initialized via `from_body_model(gender, shape)` (uses `smplx` library) or `from_verts(verts)` (bypasses shape params — useful for image-fitted meshes).
2. `measure(measurement_names)` dispatches to `measure_length` or `measure_circumference` based on `MEASUREMENT_TYPES`.
3. **Lengths** are Euclidean distances between landmark vertex indices. A landmark can be a tuple of two indices (their average is used).
4. **Circumferences** cut the mesh with a plane defined by a landmark point (origin) and a joint-to-joint vector (normal) using `trimesh.intersections.mesh_plane`. Multi-part slices are disambiguated using face segmentation (`data/smpl[x]/smpl[x]_body_parts_2_faces.json`), then the convex hull perimeter is computed.

**Key files:**
- `measurement_definitions.py` — `MEASUREMENT_TYPES`, `SMPLMeasurementDefinitions`, `SMPLXMeasurementDefinitions` (each with `LENGTHS`, `CIRCUMFERENCES`, `CIRCUMFERENCE_TO_BODYPARTS`, `possible_measurements`); also `STANDARD_LABELS` (A–P label mapping)
- `landmark_definitions.py` — `SMPL_LANDMARK_INDICES` and `SMPLX_LANDMARK_INDICES` (vertex index lookups by anatomical name)
- `joint_definitions.py` — joint name → index mappings and joint counts for each model
- `utils.py` — `filter_body_part_slices`, `convex_hull_from_3D_points`, `load_face_segmentation`, and `point_segmentation_to_face_segmentation` (utility to regenerate the face segmentation JSONs from meshcapade point segmentation files)
- `visualize.py` — `Visualizer` class using Plotly for interactive 3D rendering in browser

## Adding a New Measurement

1. Add the name and type (`LENGTH` or `CIRCUMFERENCE`) to `MEASUREMENT_TYPES` in `measurement_definitions.py`.
2. Add its definition to `SMPLMeasurementDefinitions.LENGTHS` / `CIRCUMFERENCES` (and/or `SMPLXMeasurementDefinitions`) — lengths use two landmark keys; circumferences use `{"LANDMARKS": [...], "JOINTS": [joint1, joint2]}`.
3. For circumferences: add an entry to `CIRCUMFERENCE_TO_BODYPARTS` to restrict which body-part slice is used (avoids spurious cross-sections from the plane cut).
4. Add any new landmark vertex indices to `landmark_definitions.py` if needed.

All measurements are returned in **cm**.
