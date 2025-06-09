from pathlib import Path
import cv2

video_path = Path('/capstor/scratch/cscs/lgen/datasets_aligned/rgb/hot3d/video_tar/val/P0001_b2bcbe28_24.mp4')
out_folder = Path('./example_data/P0001_b2bcbe28_24')
out_folder.mkdir(exist_ok=True)

# read video and save each frame as image

cap = cv2.VideoCapture(str(video_path))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(out_folder / f'{i:03d}.png'), frame)


