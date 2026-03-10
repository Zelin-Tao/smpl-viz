#!/usr/bin/env python3
"""
smpl-viz — visualize AMASS .npz motion files in MuJoCo.

Usage:
    smpl-viz <file.npz>
    smpl-viz <directory/>

Keybindings:
    Space        pause / resume
    ← / J        step backward one frame (paused)
    → / L        step forward one frame  (paused)
    [ / ,        slow down
    ] / .        speed up
    Q            restart clip
    N            next clip
    P            previous clip
    Esc          quit
"""
import glob
import sys
from os import path as osp

from smpl_viz.player import MotionPlayer, run_playlist


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]

    if osp.isfile(target) and target.endswith('.npz'):
        print(f"Loading: {target}")
        p = MotionPlayer(target)
        # hold_on_end=True: pause on last frame, wait for R (replay) or Q (quit)
        p.play(loop=False, hold_on_end=True)
        sys.exit(0)

    elif osp.isdir(target):
        files = sorted(glob.glob(osp.join(target, "*.npz")))
        if not files:
            print(f"No .npz files found in {target}")
            sys.exit(1)
        print(f"Found {len(files)} files in {target}")
        run_playlist(files)

    else:
        print(f"Error: '{target}' is not a .npz file or a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
