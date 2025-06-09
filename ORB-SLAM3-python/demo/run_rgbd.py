import orbslam3
import argparse
from glob import glob
import os 
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_file", default="/workspaces/Digital-Humans-Project/ORBvoc.txt")
parser.add_argument("--settings_file", default="config.yml")
parser.add_argument("--dataset_path", default="hot3d_data/train-mp4-000000/P0002_2ea9af5b_0.mp4")
args = parser.parse_args()

img_files = sorted(glob(os.path.join(args.dataset_path, 'rgb/*.png')))
slam = orbslam3.system(args.vocab_file, args.settings_file, orbslam3.Sensor.MONOCULAR)
slam.set_use_viewer(True)
slam.initialize()

for img in img_files:
    timestamp = img.split('/')[-1][:-4]
    img = cv2.imread(img, -1)
    pose = slam.process_image_mono(img, float(timestamp))
    print(pose)