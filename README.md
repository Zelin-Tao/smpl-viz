# smpl-viz

Visualize [AMASS](https://amass.is.tue.mpg.de/) motion capture `.npz` files using a SMPL-H body model rendered in MuJoCo.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Zelin-Tao/smpl-viz.git
cd smpl-viz
```

### 2. Conda environment

```bash
conda create -n smpl_viz python=3.10 -y
conda activate smpl_viz
```

### 3. Install dependencies

```bash
pip install numpy scipy mujoco
pip install -e .
```

### 4. Body models

Download **SMPL-H** from the [official website](https://mano.is.tue.mpg.de/) and place the models as follows:

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
| `Q` | Restart current clip |
| `N` | Next clip |
| `P` | Previous clip |
| `Esc` | Quit |

## Notes

- Supports both SMPL-H (156-dim poses) and SMPL-X (165-dim poses) formatted `.npz` files
- Single file mode: playback freezes on the last frame, press `Q` to replay or `Esc` to quit
- Directory mode: navigate between clips with `N` / `P`

## Python API

```python
from smpl_viz.player import MotionPlayer

p = MotionPlayer("motion.npz")
p.play(loop=True)
p.close()
```
