#!/usr/bin/env python3
"""
extract_extrinsics.py  –  estimate per‑frame camera extrinsics from an MP4.

USAGE
-----
python extract_extrinsics.py \
    --video path/to/video.mp4 \
    --voc   path/to/ORBvoc.txt \
    --yaml  path/to/camera_settings.yaml \
    [--fps 30] [--skip 0] [--out poses.csv]

Requirements
------------
* orbslam3‑python  (the wrapper you just built)
* opencv‑python    (`pip install opencv-python`)
* numpy
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import time
import matplotlib
    # or "QtAgg", "GTK3Agg", …

import matplotlib.pyplot as plt
matplotlib.use("Agg")  

import cv2
import numpy as np

try:
    import orbslam3 as orb
except ImportError as e:
    sys.exit("ORB‑SLAM3‑python is not importable – did the build succeed?\n" + str(e))




def t_SE3_to_RT(se3: np.ndarray):
    """
    Convert Sophus SE3 4×4 matrix (world‑to‑cam) into (R, t).
    Returns a 3×3 rotation matrix and a 3×1 translation vector.
    """
    R = se3[:3, :3]
    t = se3[:3, 3]
    return R, t


# ----------------------------------------------------------------------
# 1.  Helpers  ……………………………………………………………………………………
# ----------------------------------------------------------------------
def camera_centres(tcw_list):
    """
    tcw_list  – list/array of 4×4 *T_cw* (world→camera) matrices
    returns   – (N,3) array of camera centres *C_w*
    """
    C = []
    for Tcw in tcw_list:
        Rcw = Tcw[:3, :3]
        tcw = Tcw[:3,  3]
        Cw  = -Rcw.T @ tcw           # C_w = –Rᵀ·t   (inverse of R|t block)
        C.append(Cw)
    return np.vstack(C)

def rot_to_euler_zyx(R):
    """
    Convert a 3×3 rotation matrix to Z‑Y‑X (yaw‑pitch‑roll) Euler angles.
    Returns degrees as (yaw, pitch, roll).
    """
    # numerically stable test for gimbal lock
    sy = np.hypot(R[0,0], R[1,0])
    singular = sy < 1e-6

    if not singular:                      # normal case
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0],  sy)
        roll  = np.arctan2(R[2,1],  R[2,2])
    else:                                 # gimbal‑lock (|pitch|≈90°)
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0],  sy)
        roll  = 0.0

    return np.degrees([yaw, pitch, roll])

def main():
    parser = argparse.ArgumentParser(description="Estimate extrinsics for every "
                                                 "frame in an mp4 using ORB‑SLAM3.")
    parser.add_argument("--video", help="input mp4 video", default="hot3d_data/train-mp4-000000/P0002_2ea9af5b_0.mp4")
    parser.add_argument("--voc",    help="ORB‑SLAM3 vocabulary txt file", default="/workspaces/Digital-Humans-Project/ORBvoc.txt")
    parser.add_argument("--yaml", help="camera settings YAML (monocular)", default="config.yml")
    parser.add_argument("--fps",   type=float, default=None,
                        help="override video FPS if metadata is wrong")
    parser.add_argument("--skip",  type=int, default=0,
                        help="number of initial frames to skip before tracking starts")
    parser.add_argument("--out",   default=None, help="output CSV (default: <video>_extrinsics.csv)")
    args = parser.parse_args()

    vid_path = Path(args.video).expanduser().resolve()

    traj = None
    

    if not vid_path.exists():
        sys.exit(f"Video not found: {vid_path}")

    out_csv = Path(args.out or (vid_path.stem + "_extrinsics.csv")).resolve()

    # --- 1.  initialise ORB‑SLAM3 -------------------------------------------------
    slam = orb.system(
        args.voc,
        args.yaml,
        orb.Sensor.MONOCULAR,
        )
    slam.initialize()
    slam.set_use_viewer(True)
    print("[INFO] ORB‑SLAM3 initialised.")

    # --- 2.  open video -----------------------------------------------------------
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video {vid_path}")

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts_step = 1.0 / fps

    frame_idx = 0
    ts = 0.0
    poses = []             # list of (frame, timestamp, R_flattened, t)

    print("[INFO] Processing …  (Ctrl‑C to abort)")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < args.skip:
                frame_idx += 1
                ts += ts_step
                continue

            # ORB‑SLAM3 expectss grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- 3.  feed frame & grab pose --------------------------------------
            tracked = slam.process_image_mono(gray, ts)   # 4×4 Sophus matrix or None
            if tracked:
                pass
                
                    
                #print(slam.get_trajectory())
                
                  # 3×3 rotation matrix
                #R, t = t_SE3_to_RT(Tcw)
                #poses.append([frame_idx, ts, *R.flatten(), *t])
            else:
                # when tracking is lost, store NaNs
                poses.append([frame_idx, ts] + [np.nan]*15)

            frame_idx += 1
            ts += ts_step

    finally:
        cap.release()

        traj = slam.get_trajectory()  # get last trajectory
        slam.shutdown()       # flush maps, threads, viewer, etc.



    # ----------------------------------------------------------------------
    # 2.  Extract position + orientation  ……………………………………
    # ----------------------------------------------------------------------
    poses  = camera_centres(traj)                  # (N,3)
    eulers = np.array([rot_to_euler_zyx(T[:3,:3])  # (N,3)
                    for T in traj],
                    dtype=float)

    x, y, z           = poses.T
    yaw, pitch, roll  = eulers.T
    frames            = np.arange(len(traj))

    # ----------------------------------------------------------------------
    # 3‑A.  Ground‑plane trajectory (x‑z)  ………………………………………
    # ----------------------------------------------------------------------
    plt.figure()
    plt.plot(x, z, linewidth=1.0)
    plt.xlabel("x  [m]"); plt.ylabel("z  [m]")
    plt.title("Camera trajectory – ground plane (x‑z)")
    plt.axis("equal"); plt.grid(True)
    plt.savefig("traj_topdown.png", dpi=200, bbox_inches="tight")

    # ----------------------------------------------------------------------
    # 3‑B.  Full 3‑D trajectory with heading arrows  …………………
    # ----------------------------------------------------------------------
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linewidth=1.0)

    # ------------------------------------------------------------
    #  Add camera–frame triads (x/red, y/green, z/blue)
    # ------------------------------------------------------------
    triad_len = 0.005        # metres – scale as you wish
    step      = max(len(traj)//40, 1)   # plot ~40 triads max

    for i in range(0, len(traj), step):
        T   = traj[i]
        R   = T[:3, :3]               # rotation camera←world
        C   = -R.T @ T[:3, 3]         # camera centre in world

        # world‑vectors of the camera’s local +x,+y,+z axes
        x_axis, y_axis, z_axis = R.T   # columns of Rᵀ

        # one quiver per axis, colour‑coded
        ax.quiver(*C, *(triad_len*x_axis), color='r', linewidth=1)
        ax.quiver(*C, *(triad_len*y_axis), color='g', linewidth=1)
        ax.quiver(*C, *(triad_len*z_axis), color='b', linewidth=1)

    # optional: keep equal aspect & re‑draw
    ax.set_box_aspect([1,1,1])   # nicer scaling if using mpl ≥3.3
    fig.canvas.draw_idle()
    fig.savefig("traj_3d_with_triads.png", dpi=200, bbox_inches="tight")
    print("✅  Saved 3‑D plot with axis triads to traj_3d_with_triads.png")

    # … a few arrows to show orientation (“look” direction = –Rᵀ·z_cam)
    step = max(len(traj)//25, 1)          # ~25 arrows max
    for i in range(0, len(traj), step):
        R = traj[i][:3,:3]
        look = -(R.T @ np.array([0,0,1]))   # optical axis in world
        ax.quiver(x[i], y[i], z[i],
                look[0], look[1], look[2],
                length=0.05, arrow_length_ratio=0.3, linewidth=0.6)

    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]"); ax.set_zlabel("z  [m]")
    ax.set_title("Camera trajectory – 3‑D")
    ax.view_init(elev=70, azim=-90)
    fig.savefig("traj_3d.png", dpi=200, bbox_inches="tight")

    # ----------------------------------------------------------------------
    # 3‑C.  Euler‑angle evolution  ………………………………………
    # ----------------------------------------------------------------------
    plt.figure()
    plt.plot(frames, yaw,   label="yaw  (Z)", linewidth=1.0)
    plt.plot(frames, pitch, label="pitch (Y)", linewidth=1.0)
    plt.plot(frames, roll,  label="roll  (X)", linewidth=1.0)
    plt.xlabel("frame #"); plt.ylabel("angle  [deg]")
    plt.title("Orientation (ZYX Euler angles vs. frame)")
    plt.legend(); plt.grid(True)
    plt.savefig("orientation_euler.png", dpi=200, bbox_inches="tight")

    print("✅  Plots saved:\n"
        "   • traj_topdown.png\n"
        "   • traj_3d.png\n"
        "   • orientation_euler.png")
        



if __name__ == "__main__":
    main()
