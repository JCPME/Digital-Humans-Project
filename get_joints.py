import pickle
import numpy as np
from pathlib import Path

input_file = Path('/iopsstor/scratch/cscs/lgen/kaifeng/hamer/out_video/P0001_b2bcbe28_24/0/results.pkl')
with open(input_file, 'rb') as f:
    results = pickle.load(f)

num_frames = 60
left_hand = np.zeros((num_frames, 21, 3))
right_hand = np.zeros((num_frames, 21, 3))
for frame_idx in results:
    result = results[frame_idx]
    is_right = result['is_right']
    joints = result['joints']
    transl = result['transl']
    for i, right in enumerate(is_right):
        hand = joints[i]
        hand[:, 0] = hand[:, 0] * (2 * right - 1)
        # hand = hand + transl[i][None]
        if right:
            right_hand[frame_idx] = hand
        else:
            left_hand[frame_idx] = hand

save_dir = input_file.parent
np.save(save_dir / 'left_hand.npy', left_hand)
np.save(save_dir / 'right_hand.npy', right_hand)