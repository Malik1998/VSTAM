import os

import cv2
import imageio
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='join jsons from detection')
parser.add_argument('--out_dir', default="dataset/train")
parser.add_argument('--count_videos', default=10, type=int)
parser.add_argument('--frames_per_video', default=120, type=int)
parser.add_argument('--width', default=640, type=int)
parser.add_argument('--height', default=480, type=int)
parser.add_argument('--fps', default=24, type=int)


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.count_videos):
        file_name = f"{args.out_dir}/{i}.gif"
        image_lst = [np.random.rand(args.width, args.height, 3) for _ in range(args.frames_per_video)]
        imageio.mimsave(file_name, image_lst, fps=args.fps)