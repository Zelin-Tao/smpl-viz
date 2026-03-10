"""
player.py — MuJoCo-based motion player with keyboard controls.

Keybindings (active while the viewer window is focused):
  Space       pause / resume
  →  / L      step forward one frame   (when paused)
  ←  / J      step backward one frame  (when paused)
  ]  / .      speed up  (×1.25)
  [  / ,      slow down (×0.8)
  N           next clip  (folder mode)
  P           previous clip (folder mode)
  R           restart current clip from frame 0
  Q / Esc     quit
"""

import time
import threading

import mujoco
import mujoco.viewer
import numpy as np

from smpl_viz.builder import load_motion

# ---------------------------------------------------------------------------
# GLFW key codes
# ---------------------------------------------------------------------------
_KEY_SPACE = 32
_KEY_RIGHT = 262
_KEY_LEFT  = 263
_KEY_ESC   = 256
_KEY_N     = ord('N')
_KEY_P     = ord('P')
_KEY_R     = ord('R')
_KEY_Q     = ord('Q')
_KEY_J     = ord('J')
_KEY_L     = ord('L')
_KEY_LBRAK = ord('[')
_KEY_RBRAK = ord(']')
_KEY_COMMA = ord(',')
_KEY_DOT   = ord('.')


class MotionPlayer:
    """Plays a single AMASS .npz file in MuJoCo with interactive controls."""

    def __init__(self, npz_path, target_height=None):
        self.mj_model, self.mj_data, self.qpos_traj, self.base_fps = \
            load_motion(npz_path, target_height)

        self.n_frames  = self.qpos_traj.shape[0]
        self.frame_idx = 0
        self.speed     = 1.0
        self.paused    = False

        self._cmd_next = False
        self._cmd_prev = False
        self._cmd_quit = False
        self._lock     = threading.Lock()

        self._viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data,
            key_callback=self._on_key,
        )
        self._init_camera()
        self._viewer.sync()

    def _init_camera(self):
        cam = self._viewer.cam
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance  = 4.0
        cam.azimuth   = 200.0
        cam.elevation = -18.0
        cam.lookat[:] = [0.0, 0.0, 0.9]

    def _on_key(self, keycode):
        with self._lock:
            if keycode == _KEY_SPACE:
                self.paused = not self.paused

            elif keycode in (_KEY_RIGHT, _KEY_L):
                if self.paused:
                    self.frame_idx = min(self.frame_idx + 1, self.n_frames - 1)
                    self._render_frame(self.frame_idx)

            elif keycode in (_KEY_LEFT, _KEY_J):
                if self.paused:
                    self.frame_idx = max(self.frame_idx - 1, 0)
                    self._render_frame(self.frame_idx)

            elif keycode in (_KEY_RBRAK, _KEY_DOT):
                self.speed = min(self.speed * 1.25, 8.0)
                print(f"  speed: {self.speed:.2f}x")

            elif keycode in (_KEY_LBRAK, _KEY_COMMA):
                self.speed = max(self.speed * 0.8, 0.1)
                print(f"  speed: {self.speed:.2f}x")

            elif keycode == _KEY_R:
                self.frame_idx = 0
                self.paused    = False

            elif keycode == _KEY_N:
                self._cmd_next = True

            elif keycode == _KEY_P:
                self._cmd_prev = True

            elif keycode in (_KEY_Q, _KEY_ESC):
                self._cmd_quit = True

    def _render_frame(self, idx):
        self.mj_data.qpos[:] = self.qpos_traj[idx]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._viewer.sync()

    def play(self, loop=False):
        """Run the playback loop. Returns: 'next' | 'prev' | 'quit' | 'done'."""
        self.frame_idx = 0
        self.paused    = False

        while self._viewer.is_running():
            with self._lock:
                if self._cmd_quit:
                    return 'quit'
                if self._cmd_next:
                    self._cmd_next = False
                    return 'next'
                if self._cmd_prev:
                    self._cmd_prev = False
                    return 'prev'

            if not self.paused:
                t0 = time.time()
                self._render_frame(self.frame_idx)
                self.frame_idx += 1

                if self.frame_idx >= self.n_frames:
                    if loop:
                        self.frame_idx = 0
                    else:
                        return 'done'

                budget = 1.0 / (self.base_fps * self.speed)
                elapsed = time.time() - t0
                if budget > elapsed:
                    time.sleep(budget - elapsed)
            else:
                time.sleep(0.02)

        return 'quit'

    def close(self):
        self._viewer.close()


def run_playlist(npz_files, target_height=None):
    """Cycle through a list of .npz files with N / P navigation."""
    idx    = 0
    player = None

    while 0 <= idx < len(npz_files):
        path = npz_files[idx]
        print(f"[{idx + 1}/{len(npz_files)}]  {path}")
        print("  Space=pause  ←/→=step  [/]=speed  R=restart  N=next  P=prev  Q=quit")

        try:
            if player is not None:
                player.close()
            player = MotionPlayer(path, target_height)
            result = player.play(loop=False)
        except Exception as exc:
            print(f"  Error: {exc}, skipping...")
            result = 'next'

        if result == 'quit':
            break
        elif result in ('next', 'done'):
            idx += 1
        elif result == 'prev':
            idx = max(idx - 1, 0)

    if player is not None:
        player.close()
