# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation

```bash
# Basic installation
pip install xformers torch>=2 torchvision
pip install -e .

# For Gaussian Splatting support
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

# With Gradio app support (python>=3.10)
pip install -e ".[app]"

# Everything
pip install -e ".[all]"
```

### Running Inference

```bash
# CLI - auto-detect input type (image, directory, video, COLMAP)
da3 auto INPUT_PATH --export-dir ./output

# CLI - with backend (caches model in GPU for multiple jobs)
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --gallery-dir ./workspace
da3 auto INPUT_PATH --use-backend --export-dir ./output

# CLI - video processing with specific FPS
da3 video video.mp4 --fps 15 --export-format glb --export-dir ./output

# CLI - single image
da3 image image.jpg --export-dir ./output

# CLI - image directory
da3 images ./image_folder --export-dir ./output

# CLI - COLMAP dataset
da3 colmap ./colmap_data --sparse-subdir 0 --export-dir ./output

# Launch Gradio app
da3 gradio --model-dir depth-anything/DA3NESTED-GIANT-LARGE --workspace-dir ./workspace --gallery-dir ./gallery

# Launch standalone gallery server
da3 gallery --gallery-dir ./workspace
```

### DA3-Streaming (Long Video Processing)

```bash
cd da3_streaming
python da3_streaming.py --image_dir ./path_of_images --config ./configs/base_config.yaml --output_dir ${OUTPUT_DIR}
```

### Benchmark Evaluation

```bash
# Download benchmark datasets
mkdir -p workspace/benchmark_dataset
hf download depth-anything/DA3-BENCH --local-dir workspace/benchmark_dataset --repo-type dataset

# Run full evaluation
python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-GIANT

# Evaluate specific datasets/modes
python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.datasets=[hiroom] eval.modes=[pose]
```

### Python API

```python
from depth_anything_3.api import DepthAnything3
import glob

# Load model
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device="cuda")

# Run inference
images = sorted(glob.glob("assets/examples/SOH/*.png"))
prediction = model.inference(images)

# Access outputs
prediction.depth              # [N, H, W] depth maps
prediction.conf               # [N, H, W] confidence maps
prediction.extrinsics         # [N, 3, 4] camera poses (opencv/colmap format)
prediction.intrinsics         # [N, 3, 3] camera intrinsics
prediction.processed_images   # [N, H, W, 3] processed RGB images

# Export results
prediction = model.inference(
    images,
    export_dir="./output",
    export_format="mini_npz-glb",  # Multiple formats with hyphen
)
```

## Architecture Overview

### Core Model Architecture

The codebase implements Depth Anything 3 (DA3), a unified model for depth estimation, camera pose estimation, and 3D Gaussian Splatting.

**Main Model Components** (`src/depth_anything_3/model/`):
- `da3.py`: `DepthAnything3Net` - Main model class with backbone, head, camera decoders
- `dinov2/`: DinoV2 vision transformer backbone
- `dpt.py`: Standard DPT head for depth prediction
- `dualdpt.py`: Dual DPT head for depth+ray prediction
- `gsdpt.py`: DPT variant for 3D Gaussian Splatting
- `cam_enc.py`/`cam_dec.py`: Camera encoder/decoder for pose estimation
- `gs_adapter.py`: Adapter for 3D Gaussian output
- `reference_view_selector.py`: Selects optimal reference view for multi-view inputs

**Model Types**:
- **Any-view models** (da3-giant, da3-large, da3-base, da3-small): Full pipeline with depth, pose, and optional GS
- **Monocular metric** (da3metric-large): Metric depth with sky segmentation
- **Monocular relative** (da3mono-large): Relative depth only
- **Nested models** (da3nested-giant-large): Combines any-view + metric models for metric-scale output

### Configuration System

Models are configured via YAML files in `src/depth_anything_3/configs/` using a declarative `__object__` syntax:

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params
  name: vitb
  out_layers: [5, 7, 9, 11]

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params
  dim_in: 1536
  output_dim: 2  # depth + ray
```

Configurations support inheritance via `__inherit__` key (see `da3nested-giant-large.yaml`).

### Input/Output Processing

**Input Processing** (`src/depth_anything_3/utils/io/input_processor.py`):
- Handles images (file paths, numpy arrays, PIL Images)
- Resize methods: `upper_bound_resize`, `lower_bound_resize`
- Default processing resolution: 504px

**Output Processing** (`src/depth_anything_3/utils/io/output_processor.py`):
- Handles pose alignment (Umeyama)
- Metric depth scaling (for nested models)
- Sky region handling

### Export Formats

Located in `src/depth_anything_3/utils/export/`:
- `npz.py`: `mini_npz`, `npz` - Compressed depth data
- `glb.py`: `glb` - 3D scene with point cloud and cameras
- `gs.py`: `gs_ply`, `gs_video` - 3D Gaussian Splatting exports
- `feat_vis.py`: `feat_vis` - Feature visualization
- `depth_vis.py`: `depth_vis` - Depth visualization
- `colmap.py`: COLMAP format export

Multiple formats can be combined with hyphens: `mini_npz-glb-feat_vis`.

### CLI and Services

- `cli.py`: Main CLI entry point with Typer
  - Input type auto-detection (image, images, video, colmap)
  - Backend service for GPU model caching
- `services/backend.py`: FastAPI backend service
- `services/inference_service.py`: Inference request handling
- `services/gallery.py`: Gallery browser for exported scenes
- `app/gradio_app.py`: Gradio web interface

### DA3-Streaming Submodule

Located in `da3_streaming/`:
- Memory-efficient chunked inference for long videos
- `da3_streaming.py`: Main streaming pipeline
- `configs/`: YAML configs for different datasets/scenes
- Handles chunking, overlap, loop closure

### Key Utilities

- `geometry.py`: 3D transformations, coordinate conversions
- `pose_align.py`: Umeyama alignment for pose conditioning
- `alignment.py`: Metric scaling, sky masks
- `ray_utils.py`: Ray-to-pose conversion
- `camera_trj_helpers.py`: Camera trajectory generation for GS video

### Model Registry

Available models are registered in `src/depth_anything_3/registry.py`:
- `da3-giant`: 1.15B params, any-view with GS
- `da3-large`: 0.35B params, any-view (recommended)
- `da3-base`: 0.12B params, any-view
- `da3-small`: 0.08B params, any-view
- `da3metric-large`: 0.35B params, metric depth
- `da3mono-large`: 0.35B params, monocular relative depth
- `da3nested-giant-large`: 1.40B params, nested (any-view + metric)

## Important Notes

- Models use **xformers** for memory-efficient attention
- Gaussian Splatting requires `gsplat` from specific commit
- For metric depth from DA3METRIC-LARGE: `metric_depth = focal * net_output / 300.`
- Nested models (DA3NESTED) output metric depth directly without scaling
- `use_ray_pose=True` uses ray-based pose estimation (slower but more accurate)
- Reference view selection (`ref_view_strategy`) matters for multi-view inputs
- DA3-Streaming configs have specific parameters for chunk size, overlap, GPU optimization

## Dataset Paths

Benchmark datasets expected in `workspace/benchmark_dataset/` (configurable in `utils/constants.py`):
- `eth3d/`: ETH3D multi-view stereo
- `7scenes/`: RGB-D indoor localization
- `scannetpp/`: High-quality indoor RGB-D
- `hiroom/data/`: High-resolution indoor rooms
- `dtu/`: DTU multi-view stereo (49 views)
- `dtu64/`: DTU pose subset (64 views)
