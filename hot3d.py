import pickle
import numpy as np
from pathlib import Path
import json
import pdb
from tqdm import tqdm
import statistics as st
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter
from scipy.signal import filtfilt
import os

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


import numpy as np

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Applies Butterworth low-pass filter along time (axis=1).
    
    :param data: np.ndarray of shape [2, 60, 21, 3]
    :param cutoff: desired cutoff frequency (Hz)
    :param fs: sampling rate (Hz)
    :param order: filter order
    :return: filtered data
    """
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = np.zeros_like(data)
    
    # Apply filter to each [21,3] pair over time
    for i in range(data.shape[0]):  # for each sample
        for j in range(data.shape[2]):  # for each joint/feature
            for k in range(data.shape[3]):  # for x/y/z or features
                filtered[i, :, j, k] = filtfilt(b, a, data[i, :, j, k])
    
    return filtered

def hand_selection(results):
    frames = sorted(results.keys())

    #### find anchor frame ####a
    valid = [f for f in frames if len(results[f]['is_right'])>=1] #finds all frames with at least one hand

    two_hands_found = False
    i = 0
    anchor_frame = 0

    while two_hands_found == False and i < len(valid):
        allhands = results[valid[i]]['is_right']
        all_one_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 1]
        all_zero_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 0] 
        if len(all_one_idx) > 0 and len(all_zero_idx) > 0:
            two_hands_found = True
            anchor_frame = valid[i]
        else:
            i = i+1
    if two_hands_found == False:
            return results
    allhands = results[valid[i]]['is_right']

    ### last hands ###
    # last_zero_idx = next(i for i, v in reversed(list(enumerate(allhands))) if v == 0)
    # last_one_idx  = next(i for i, v in reversed(list(enumerate(allhands))) if v == 1)
    

    ### first hands ###
    last_zero_idx = next(i for i, v in enumerate(allhands) if v == 0)
    last_one_idx  = next(i for i, v in enumerate(allhands) if v == 1)

    l_hand = results[anchor_frame]['joints'][last_zero_idx]
    l_transl = results[anchor_frame]['transl'][last_zero_idx]
    r_hand = results[anchor_frame]['joints'][last_one_idx]
    r_transl = results[anchor_frame]['transl'][last_one_idx]

    results[anchor_frame]['joints'] = np.array([l_hand,r_hand])
    results[anchor_frame]['is_right'] = np.array([0.,1.],dtype=np.float64)
    results[anchor_frame]['transl'] = np.array([l_transl,r_transl])
    
    anchor_frame_idx = frames.index(anchor_frame)
    #### extract all the hands and transl####
    allright = []
    alltransl_r = []
    allleft = []
    alltransl_l = []
    allrotl = []
    allrotr = []
    for frame in frames:
        allright_current = []
        alltransl_r_current = []
        allleft_current = []
        alltransl_l_current = []
        allrotl_current = []
        allrotr_current = []
        for hand_idx in range(len(results[frame]['is_right'])):      #iterates over all hands 
            if results[frame]['is_right'][hand_idx] == 1:
               allright_current.append(results[frame]['joints'][hand_idx])
               alltransl_r_current.append(results[frame]['transl'][hand_idx])
               allrotr_current.append(results[frame]['mano_params']['global_orient'][hand_idx])
            elif results[frame]['is_right'][hand_idx] == 0:
               allleft_current.append(results[frame]['joints'][hand_idx])
               alltransl_l_current.append(results[frame]['transl'][hand_idx])
               allrotl_current.append(results[frame]['mano_params']['global_orient'][hand_idx])

        if len(allright_current)>0:
            allright_current = np.stack(allright_current, axis=0)
            alltransl_r_current = np.stack(alltransl_r_current, axis=0)
            allrotr_current = np.stack(allrotr_current,axis = 0)
            allright.append(allright_current)
            alltransl_r.append(alltransl_r_current)
            allrotr.append(allrotr_current)
        elif len(allright_current) == 0:
            allright.append(np.full((1,21,3), np.nan))
            alltransl_r.append(np.full((1,3), np.nan))
            allrotr.append(np.full((1,1,3,3), np.nan))
        if len(allleft_current)>0:
            allleft_current = np.stack(allleft_current, axis=0)
            alltransl_l_current = np.stack(alltransl_l_current, axis=0)
            allrotl_current = np.stack(allrotl_current, axis = 0)
            allleft.append(allleft_current)
            alltransl_l.append(alltransl_l_current)
            allrotl.append(allrotl_current)
        elif len(allleft_current) == 0:
            allleft.append(np.full((1,21,3), np.nan))
            alltransl_l.append(np.full((1,3), np.nan))
            allrotl.append(np.full((1,1,3,3), np.nan))
    #### correct at anchor frame ####
    allright[anchor_frame_idx] = r_hand.reshape(1,21,3)
    alltransl_r[anchor_frame_idx] = r_transl.reshape(3,)
    allleft[anchor_frame_idx] = l_hand.reshape(1,21,3)
    alltransl_l[anchor_frame_idx] = l_transl.reshape(3,)

    if anchor_frame_idx>0:
        for i in range(anchor_frame_idx-1, -1, -1):
            if np.isnan(allright[i]).any():
                allright[i] = allright[i+1]
                alltransl_r[i] = alltransl_r[i+1]
                allrotr[i] = allrotr[i+1]
            else:
                r_diffs = allright[i+1][0]-allright[i][:]
                r_dist = np.linalg.norm(r_diffs,axis=2)
                r_dist = np.abs(np.mean(r_dist,axis=1))
                closest_idx = np.argmin(r_dist)

                allright[i] = allright[i][closest_idx]
                allright[i] = allright[i].reshape(1,21,3)
                alltransl_r[i] = alltransl_r[i][closest_idx]
                allrotr[i] = allrotr[i][closest_idx]
                #alltransl_r[i] = alltransl_r[i].reshape(1,3)
    ## right hand forward ##
    if anchor_frame_idx<len(frames)-1:
        for i in range(anchor_frame_idx+1,len(frames)):
            if np.isnan(allright[i]).any():
                allright[i] = allright[i-1]
                alltransl_r[i] = alltransl_r[i-1]
                allrotr[i] = allrotr[i-1]
            else:
                r_diffs = allright[i-1][0] - allright[i][:]
                r_dist = np.linalg.norm(r_diffs,axis=2)
                r_dist = np.abs(np.mean(r_dist,axis=1))
                closest_idx = np.argmin(r_dist)
                #print(r_diffs.shape,closest_idx)
                allright[i] = allright[i][closest_idx]
                allright[i] = allright[i].reshape(1,21,3)
                alltransl_r[i] = alltransl_r[i][closest_idx]
                allrotr[i] = allrotr[i][closest_idx]
                #alltransl_r[i] = alltransl_r[i].reshape(1,3)

    ## left hand backward ##
    if anchor_frame_idx>0:
        for i in range(anchor_frame_idx-1, -1, -1):
            if np.isnan(allleft[i]).any():
                allleft[i] = allleft[i+1]
                alltransl_l[i] = alltransl_l[i+1]
                allrotl[i] = allrotl[i+1]
            else:
                l_diffs = allleft[i+1][0] - allleft[i][:]
                l_dist = np.linalg.norm(l_diffs,axis=2)
                l_dist = np.abs(np.mean(l_dist,axis=1))
                closest_idx = np.argmin(l_dist)

                allleft[i] = allleft[i][closest_idx]
                allleft[i] = allleft[i].reshape(1,21,3)
                alltransl_l[i] = alltransl_l[i][closest_idx]
                allrotl[i] = allrotl[i][closest_idx]
                #alltransl_l[i] = alltransl_l[i].reshape(1,3)
    ## left hand forward ##
    if anchor_frame_idx<len(frames)-1:
        for i in range(anchor_frame_idx+1,len(frames)):
            if np.isnan(allleft[i]).any():
                allleft[i] = allleft[i-1]
                alltransl_l[i] = alltransl_l[i-1]
                allrotl[i] = allrotl[i-1]
            else:
                l_diffs = allleft[i-1][0]-allleft[i][:]
                l_dist = np.linalg.norm(l_diffs,axis=2)
                l_dist = np.abs(np.mean(l_dist,axis=1))
                closest_idx = np.argmin(l_dist)
            #print(allleft[i].shape,frames[i])
                allleft[i] = allleft[i][closest_idx]
                allleft[i] = allleft[i].reshape(1,21,3)
                alltransl_l[i] = alltransl_l[i][closest_idx]
                allrotl[i] = allrotl[i][closest_idx]
                #alltransl_l[i] = alltransl_l[i].reshape(1,3)
    #print(f"Anchor Frame: {anchor_frame}",len(frames))
    for i in range(len(frames)):
        frame = frames[i]
        results[frame]['joints'] = np.array([allleft[i][0],allright[i][0]])
        
        #print(alltransl_l[i].shape,alltransl_r[i].shape,i)
        alltransl_l[i] = alltransl_l[i].reshape(3,)
        alltransl_r[i] = alltransl_r[i].reshape(3,)
        allrotr[i] = allrotr[i].reshape(1,3,3)
        allrotl[i] = allrotl[i].reshape(1,3,3)
        results[frame]['transl'] = np.array([alltransl_l[i],alltransl_r[i]])
        results[frame]['is_right'] = np.array([0.,1.],dtype=np.float64)
        results[frame]['mano_params']['global_orient'] = np.array([allrotl[i],allrotr[i]])
        assert results[frame]['transl'].shape == (2,3),f"wrong dimension at position {i}"
    
    #for frame in frames:
        #print(results[frame]['joints'].shape,results[frame]['transl'].shape,results[frame]['is_right'].shape,frame)
        #print(allright[frame].shape,allleft[frame].shape,alltransl_r[frame].shape,alltransl_l[frame].shape)
    return results

def altbase(results):
    frames = sorted(results.keys())
    valid = [f for f in frames if len(results[f]['is_right'])>=1] #finds all frames with at least one hand

    two_hands_found = False
    i = 0
    anchor_frame = 0

    while two_hands_found == False and i < len(valid):
        allhands = results[valid[i]]['is_right']
        all_one_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 1]
        all_zero_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 0] 
        if len(all_one_idx) > 0 and len(all_zero_idx) > 0:
            two_hands_found = True
            anchor_frame = valid[i]
        else:
            i = i+1
    if two_hands_found == False:
            return results
    
    anchor_idx = frames.index(anchor_frame)
    if anchor_idx>0:
        for i in range(anchor_idx-1,-1,-1):
            frame = frames[i]
            next_frame = frames[i+1]
            allhands = results[frame]['is_right']
            next_allhands = results[next_frame]['is_right']
            all_one_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 1]
            all_zero_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 0] 
            last_zero_idx = next(i for i, v in enumerate(next_allhands) if v == 0)
            last_one_idx  = next(i for i, v in enumerate(next_allhands) if v == 1)
            if len(all_one_idx) == 0:
                results[frame]['joints'] = np.vstack([results[frame]['joints'],results[next_frame]['joints'][last_one_idx].reshape(1,21,3)])
                results[frame]['transl'] = np.vstack([results[frame]['transl'],results[next_frame]['transl'][last_one_idx].reshape(1,3)])
                results[frame]['is_right'] = np.append(results[frame]['is_right'],1)
            if len(all_zero_idx) == 0:
                results[frame]['joints'] = np.vstack([results[frame]['joints'],results[next_frame]['joints'][last_zero_idx].reshape(1,21,3)])
                results[frame]['transl'] = np.vstack([results[frame]['transl'],results[next_frame]['transl'][last_zero_idx].reshape(1,3)])
                results[frame]['is_right'] = np.append(results[frame]['is_right'],0)
                
    if anchor_idx <len(frames):
        for i in range(anchor_idx+1,len(frames)):
            frame = frames[i]
            prev_frame = frames[i-1]
            allhands = results[frame]['is_right']
            prev_allhands = results[prev_frame]['is_right']
            all_one_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 1]
            all_zero_idx = [i for i, v in reversed(list(enumerate(allhands))) if v == 0] 
            last_zero_idx = next(i for i, v in enumerate(prev_allhands) if v == 0)
            last_one_idx  = next(i for i, v in enumerate(prev_allhands) if v == 1)
            if len(all_one_idx) == 0:
                results[frame]['joints'] = np.vstack([results[frame]['joints'],results[prev_frame]['joints'][last_one_idx].reshape(1,21,3)])
                results[frame]['transl'] = np.vstack([results[frame]['transl'],results[prev_frame]['transl'][last_one_idx].reshape(1,3)])
                results[frame]['is_right'] = np.append(results[frame]['is_right'],1)
            if len(all_zero_idx) == 0:
                results[frame]['joints'] = np.vstack([results[frame]['joints'],results[prev_frame]['joints'][last_zero_idx].reshape(1,21,3)])
                results[frame]['transl'] = np.vstack([results[frame]['transl'],results[prev_frame]['transl'][last_zero_idx].reshape(1,3)])
                results[frame]['is_right'] = np.append(results[frame]['is_right'],0)
    return results

def z_corr1(results, K, win):
    frames = sorted(results.keys())
    alltransl_L = []
    alltransl_R = []
    if len(frames)<K:
        return results
    for i in range(0,K):
        frame = frames[i]
        is_right = results[frame]['is_right']
        transl = results[frame]['transl']
        for h in range(len(is_right)):
            if is_right[h] == 0:
                alltransl_L.append(transl[h])
            elif is_right[h] == 1:
                alltransl_R.append(transl[h])
            else:
                return results
    if len(alltransl_L) == 0 or len(alltransl_R) == 0:
        return results
  
    anchor_L = np.vstack(alltransl_L).mean(0)
    anchor_R = np.vstack(alltransl_R).mean(0)
    beta_hist_l = []
    beta_hist_r = []
    #raw_z = { f: results[f]['transl'][:, 2].copy() for f in frames }

    for i in range(len(frames)):
        frame = frames[i]
        is_right = results[frame]['is_right']
        for h in range(len(is_right)):
            z = results[frame]['transl'][h][2]
            if is_right[h]==0:
                beta = anchor_L[2]-z#raw_z[frame][h]
                beta_hist_l.append(beta)
                if len(beta_hist_l)>win:
                    beta_hist_l.pop(0)
                beta_mean = np.mean(beta_hist_l)

         
            elif is_right[h]==1:
                beta = anchor_R[2]-z#raw_z[frame][h]
                beta_hist_r.append(beta)
                if len(beta_hist_r)>win:
                    beta_hist_r.pop(0)
                beta_mean = np.mean(beta_hist_r)

            results[frame]['transl'][h][2] = 0.3*results[frame]['transl'][h][2]+0.7*beta_mean
    return results
def z_corr(results):
    frames = sorted(results.keys())
    
    anchors_l = []
    anchors_r = []
    K=10
    while K>len(frames):
        K-=1    

    for i in range(K):
        f = frames[i]
        allhands = results[f]['is_right']
        transl = results[f]['transl']
        for h in range(len(allhands)):
            if allhands[h] == 0.0:
                anchors_l.append(transl[h])
            elif allhands[h] == 1.0:
                anchors_r.append(transl[h])
            else:
                print(allhands[h], K)
                return results, 0.5
    z_anchor_l = np.mean(anchors_l,axis=0)
    z_anchor_r = np.mean(anchors_r,axis=0)
    
    for f in frames:
        allhands = results[f]['is_right']
        transl = results[f]['transl']
        rotmat = results[f]['mano_params']['global_orient']
        wrist = results[f]['joints']
        for h in range(len(allhands)):
            if allhands[h] == 0:
                z_pred = transl[h] + rotmat[h][0] @ wrist[h][0]
                beta = z_pred - z_anchor_l
                z_corr = z_pred - beta
                results[f]['transl'][h] = z_corr
            elif allhands[h] == 1:
                z_pred = transl[h,-1] + rotmat[h][0] @ wrist[h][0]
                beta = z_pred - z_anchor_r
                z_corr = z_pred - beta
                results[f]['transl'][h] = z_corr
    z_anchor = (z_anchor_l + z_anchor_r)/2

    return results, z_anchor

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

def process_hamer(nancount, seq_path, save_joints=False, use_slam = False):
    segment_id = int(seq_path.parent.name)
    seq_id = seq_path.parent.parent.name
    
    num_frames = 60
    start_frame = segment_id * num_frames
    end_frame = start_frame + num_frames
    valid_hands = np.ones((2, num_frames), dtype=bool)
    
    gt_dir = Path('./hot3d_data/known_val')
    gt_labels = np.load(gt_dir / f'{seq_id}.npz')
    gt_lhand = gt_labels['lhand'][start_frame:end_frame]  # [60, 21, 3]

    if use_slam:
        slam_path = Path('./slam_out')
        slam_file = os.path.join(slam_path,f'{seq_id}.pkl')

        if not os.path.exists(slam_file):
            print(f'SLAM file {slam_file} not found')
            return None

        with open(slam_file, 'rb') as file:
            data = pickle.load(file)
            gt_cam=data
    else:
        gt_cam = gt_labels['cam'][start_frame:end_frame]  # [60, 4, 4] 
    
    


    # if any joint of one frame is nan, set that frame to invalid
    for i in range(num_frames):
        if np.isnan(gt_lhand[i]).any():
            valid_hands[0, i] = False
    # check nan, inf and replace with 0
    if np.isnan(gt_lhand).any() or np.isinf(gt_lhand).any():
        nancount = nancount + 1
        print(f'Found nan or inf in {seq_id} lhand')
        gt_lhand = np.nan_to_num(gt_lhand)
    gt_rhand = gt_labels['rhand'][start_frame:end_frame]
    for i in range(num_frames):
        if np.isnan(gt_rhand[i]).any():
            valid_hands[1, i] = False
    if np.isnan(gt_rhand).any() or np.isinf(gt_rhand).any():
        nancount = nancount + 1
        print(f'Found nan or inf in {seq_id} rhand')
        gt_rhand = np.nan_to_num(gt_rhand)
    gt_hand_joints = np.stack([gt_lhand, gt_rhand])
                     ####
    gt_hand_visible = gt_labels['hand_visible'][start_frame:end_frame]  # [60, ]
    
    left_hand_joints = np.zeros((num_frames, 21, 3))
    right_hand_joints = np.zeros((num_frames, 21, 3))
    has_pred = np.zeros((2, num_frames), dtype=bool)
    with open(seq_path, 'rb') as f:
        results = pickle.load(f)

    results = altbase(results)              #fills missing hands (disable for validation)
    
    #results = hand_selection(results)      #select closest hand and fills missing hands
    #results = z_corr1(results,10,1)        #correct depth estimation
   
    
    # first frames where left and right hand has prediction
    first_pred_frame = np.ones(2, dtype=int) * (num_frames - 1)
    for frame_idx in results:
        result = results[frame_idx]
        is_right = result['is_right'].astype(np.int32)
        joints = result['joints']
        joints = joints[:, openpose_to_mano]
        transl = result['transl']
        #transl[:,-1] = 0
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
    pred_hand_joints = butter_lowpass_filter(pred_hand_joints, cutoff=0.1, fs=30, order=2)      #apply butterworth filter
    # print('pred', pred_hand_joints[:, 0])
    valid_hands = valid_hands & gt_hand_visible[None, :]
    valid_hands = valid_hands & has_pred  # [2, 60]
    
    # use camera to world transformation to transform hand joints to world space
    #rotation = gt_cam[0]
    #translation = gt_cam[1]

    rotation = gt_cam[:,:3,:3]
    translation = gt_cam[:,:3,3]
    pred_hand_joints_world = np.zeros((2, num_frames, 21, 3))
    for i in range(2):
        for j in range(num_frames):
                rotation[j] = rotation[j].T
                translation[j] = -rotation[j]@translation[j]
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
    return metrics,nancount

metrics_all = None
nancount = 0
results_dir = Path('./out_video')
result_path_list = list(results_dir.glob('./*/*/results.pkl'))
for result_path in tqdm(result_path_list):
    metrics,nancount = process_hamer(nancount, result_path, save_joints=True, use_slam=False)
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
print(f'Amount of NaNs: {nancount}')
with open('metrics.json', 'w') as f:
    json.dump(metrics_all, f)
