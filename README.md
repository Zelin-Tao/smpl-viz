# smpl-viz

Visualize [AMASS](https://amass.is.tue.mpg.de/) motion capture `.npz` files using a SMPL-H body model rendered in MuJoCo.

## Prerequisites

### 1. Conda environment

```bash
conda create -n smpl_viz python=3.10 -y
conda activate smpl_viz
```

### 2. Install dependencies

```bash
pip install numpy scipy mujoco
```

### 3. Install smpl-viz

```bash
pip install -e .
```

### 4. Body models

Download **SMPL-H** from the official website and place the models as follows:

```
~/.amass_data/
└── body_models/
    └── smplh/
        ├── male/model.npz
        ├── female/model.npz
        └── neutral/model.npz
```

To use a custom path:
```bash
export AMASS_DATA_PATH=/path/to/your/amass_data
```

## Usage

```bash
conda activate smpl_viz

# Single file
smpl-viz /path/to/motion.npz

# Entire directory
smpl-viz /path/to/motions/
```

## Keybindings

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `← / J` | Step backward one frame (paused) |
| `→ / L` | Step forward one frame (paused) |
| `[ / ,` | Slow down |
| `] / .` | Speed up |
| `R` | Restart current clip |
| `N` | Next clip |
| `P` | Previous clip |
| `Q / Esc` | Quit |

## Python API

```python
from smpl_viz.player import MotionPlayer

p = MotionPlayer("motion.npz")
p.play(loop=True)
p.close()
```