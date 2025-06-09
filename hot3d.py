import pickle
import numpy as np
from pathlib import Path
import json
import pdb
from tqdm import tqdm

import torch
from torch import nn 
from smplx import MANO

skeleton = np.array([
        [0, 13, 14, 15, 16], [0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20]
    ])
mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
openpose_to_mano = np.zeros(21, dtype=np.int32)
for i, j in enumerate(mano_to_openpose):
    openpose_to_mano[j] = i


class SimpleKalman:
    def __init__(self, dt=1/30., q=1e-2, r=1e-1):
        self.dt=dt
        self.F = np.array([[1,0,0,dt,0,0],
                           [0,1,0,0,dt,0],
                           [0,0,1,0,0,dt],
                           [0,0,0,1,0,0],
                           [0,0,0,0,1,0],
                           [0,0,0,0,0,1]])
        
        self.H = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0],
                           [0,0,1,0,0,0]])
        
        self.Q = np.eye(6) * q
        self.R = np.eye(3) * r
    def predict(self):
        self.x = np.zeros((6,))       # [x,y,z,vx,vy,vz]
        self.P = np.eye(6) * 1000.    # große anfängliche Unsicherheit
    def update(self,z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman-Gain

        # Update
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - K @ self.H) @ self.P
    def current_position(self):
        return self.x[:3]
"""    
def improve(results,dt=1/30., q=1e-2, r=1e-1):
    # 1) Load
    T = len(results)
    nhands, njoints, _ = results[0]['joints'].shape

    # 2) Kalman-Filter initialisieren mit erstem Messwert
    filters = []
    first = results[0]['joints']
    for h in range(nhands):
        hand_filters = []
        for j in range(njoints):
            init_z = first[h, j]
            kf = SimpleKalman(dt=dt, q=q, r=r, init_z=init_z)
            hand_filters.append(kf)
        filters.append(hand_filters)

    # 3) Speicher für geglättete Daten anlegen
    for frame in results:
        frame['joints'] = np.zeros((nhands, njoints, 3), dtype=float)

    # 4) Filter durchziehen
    for t in range(T):
        meas = results[t]['joints']  # Shape (2,21,3)
        for h in range(nhands):
            for j in range(njoints):
                kf = filters[h][j]
                kf.predict()
                kf.update(meas[h, j])
                results[t]['joints_smoothed'][h, j] = kf.position
    return results
"""    

def canonicalize(seq, canonical_frame=0):
    # seq: 60 x 21 x 3 for a motion seq. idx 4: middle mcp. idx 0: wrist
    # print(seq.shape)
    # Get first frame's wrist and MCP
    first_jts = seq[canonical_frame]
    wrist = first_jts[0]
    middle_mcp = first_jts[4]

    # check if the first frame is valid
    if np.all(wrist == middle_mcp) or np.isnan(first_jts).any():
        return seq

    # Compute Y-axis (projected to XY-plane)
    y_axis = middle_mcp - wrist
    y_axis[2] = 0  # Flatten
    if np.linalg.norm(y_axis) < 1e-6:
        pdb.set_trace()
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Compute orthonormal X-axis (cross(Y, Z))
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # [X, Y, Z] as columns (canonical -> world)
    rot_canonical_2_world = np.stack([x_axis, y_axis, z_axis], axis=1)

    translated = seq - wrist
    canonical_seq = translated @ rot_canonical_2_world

    return canonical_seq

def process_hamer(seq_path, save_joints=False):
    segment_id = int(seq_path.parent.name)
    seq_id = seq_path.parent.parent.name
    
    num_frames = 60
    start_frame = segment_id * num_frames
    end_frame = start_frame + num_frames
    valid_hands = np.ones((2, num_frames), dtype=bool)
    
    gt_dir = Path('./hot3d_data/label')
    gt_labels = np.load(gt_dir / f'{seq_id}.npz')
    gt_lhand = gt_labels['lhand'][start_frame:end_frame]  # [60, 21, 3]
    # if any joint of one frame is nan, set that frame to invalid
    for i in range(num_frames):
        if np.isnan(gt_lhand[i]).any():
            valid_hands[0, i] = False
    # check nan, inf and replace with 0
    if np.isnan(gt_lhand).any() or np.isinf(gt_lhand).any():
        print(f'Found nan or inf in {seq_id} lhand')
        gt_lhand = np.nan_to_num(gt_lhand)
    gt_rhand = gt_labels['rhand'][start_frame:end_frame]
    for i in range(num_frames):
        if np.isnan(gt_rhand[i]).any():
            valid_hands[1, i] = False
    if np.isnan(gt_rhand).any() or np.isinf(gt_rhand).any():
        print(f'Found nan or inf in {seq_id} rhand')
        gt_rhand = np.nan_to_num(gt_rhand)
    gt_hand_joints = np.stack([gt_lhand, gt_rhand])
    gt_cam = gt_labels['cam'][start_frame:end_frame]  # [60, 4, 4]
    gt_hand_visible = gt_labels['hand_visible'][start_frame:end_frame]  # [60, ]
    
    left_hand_joints = np.zeros((num_frames, 21, 3))
    right_hand_joints = np.zeros((num_frames, 21, 3))
    has_pred = np.zeros((2, num_frames), dtype=bool)
    with open(seq_path, 'rb') as f:
        results = pickle.load(f)

    '''
    with open(r'out_video\P0002_2ea9af5b_0\1\results.pkl','rb') as f:
        results = pickle.load(f)    

    results = improve(results)
    '''
    
    # first frames where left and right hand has prediction
    first_pred_frame = np.ones(2, dtype=int) * (num_frames - 1)
    for frame_idx in results:
        result = results[frame_idx]
        is_right = result['is_right'].astype(np.int32)
        joints = result['joints']
        joints = joints[:, openpose_to_mano]
        transl = result['transl']
        frame_idx = int(frame_idx)
        for i, right in enumerate(is_right):
            # print(right, frame_idx)
            has_pred[right, frame_idx] = True
            hand_joints = joints[i]
            hand_joints[:, 0] = hand_joints[:, 0] * (2 * right - 1)
            hand_joints = hand_joints + transl[i][None]
            if right:
                if frame_idx < first_pred_frame[1]:
                    first_pred_frame[1] = frame_idx
                # print(f'frame {frame_idx} right transl {transl[i]}')
                right_hand_joints[frame_idx] = hand_joints                
            else:
                if frame_idx < first_pred_frame[0]:
                    first_pred_frame[0] = frame_idx
                # print(f'frame {frame_idx} left transl {transl[i]}')
                left_hand_joints[frame_idx] = hand_joints

    pred_hand_joints = np.stack([left_hand_joints, right_hand_joints])  # [2, 60, 21, 3]
    # print('pred', pred_hand_joints[:, 0])
    valid_hands = valid_hands & gt_hand_visible[None, :]
    valid_hands = valid_hands & has_pred  # [2, 60]
    
    # use camera to world transformation to transform hand joints to world space
    rotation = gt_cam[:, :3, :3]
    translation = gt_cam[:, :3, 3]
    pred_hand_joints_world = np.zeros((2, num_frames, 21, 3))
    for i in range(2):
        for j in range(num_frames):
                pred_hand_joints_world[i, j] = pred_hand_joints[i, j] @ rotation[j].T + translation[j]
    
    # canonicalize
    pred_canonical = np.zeros((2, num_frames, 21, 3))
    gt_canonical = np.zeros((2, num_frames, 21, 3))
    for i in range(2):
        # print('canonicalize pred')
        pred_canonical[i] = canonicalize(pred_hand_joints_world[i], canonical_frame=first_pred_frame[i])
        # print('canonicalize gt')
        gt_canonical[i] = canonicalize(gt_hand_joints[i], canonical_frame=first_pred_frame[i])
    
    pred_canonical_perframe = np.zeros((2, num_frames, 21, 3))
    gt_canonical_perframe = np.zeros((2, num_frames, 21, 3))
    for i in range(2):
        for j in range(num_frames):
            pred_canonical_perframe[i, j] = canonicalize(pred_hand_joints_world[i, [j]])[0]
            gt_canonical_perframe[i, j] = canonicalize(gt_hand_joints[i, [j]])[0]

    #  calculate mean per joint error
    joint_error = (pred_canonical - gt_canonical) ** 2
    joint_error = np.sqrt(joint_error.sum(axis=-1))  # [2, 60, 21]
    joint_error = joint_error[valid_hands]  # [num_valid_hands, 21]
    joint_error = joint_error.mean()

    joint_error_perframe = (pred_canonical_perframe - gt_canonical_perframe) ** 2
    joint_error_perframe = np.sqrt(joint_error_perframe.sum(axis=-1))  # [2, 60, 21]
    joint_error_perframe = joint_error_perframe[valid_hands]  # [num_valid_hands, 21]
    joint_error_perframe = joint_error_perframe.mean()  # [21]

    if save_joints:
        np.save(seq_path.parent / 'pred.npy', pred_hand_joints)
        np.save(seq_path.parent / 'pred_world.npy', pred_hand_joints_world)
        np.save(seq_path.parent / 'gt.npy', gt_hand_joints)
        np.save(seq_path.parent / f'pred_canonical_{first_pred_frame[0]}_{first_pred_frame[1]}.npy', pred_canonical)
        np.save(seq_path.parent / f'gt_canonical_{first_pred_frame[0]}_{first_pred_frame[1]}.npy', gt_canonical)

    metrics = {
        'joint_error': joint_error,
        'joint_error_perframe': joint_error_perframe
    }
    return metrics

metrics_all = None

results_dir = Path('./out_video')
result_path_list = list(results_dir.glob('./*/*/results.pkl'))
for result_path in tqdm(result_path_list):
    metrics = process_hamer(result_path, save_joints=True)
    print("hammering")
    if metrics_all is None:
        metrics_all = {}
        for key in metrics:
            metrics_all[key] = []
    for key in metrics_all:
        if not np.isnan(metrics[key]):
            metrics_all[key].append(metrics[key])

for key in metrics_all:
    metrics_all[key] = np.mean(metrics_all[key])
print(metrics_all)
with open('metrics.json', 'w') as f:
    json.dump(metrics_all, f)
