import os

import cv2
import imageio
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='create random masks')
parser.add_argument('--out_dir', default="dataset/train_mask")
parser.add_argument('--count_videos', default=10, type=int)
parser.add_argument('--frames_per_video', default=120, type=int)
parser.add_argument('--width', default=640, type=int)
parser.add_argument('--height', default=480, type=int)
parser.add_argument('--fps', default=24, type=int)


def create_random_mask(width, height, count_elements):
    mask = np.zeros((width, height, 3), dtype=np.uint8)
    for _ in range(count_elements):
        w = np.random.randint(width // 4, width // 2)
        h = np.random.randint(height // 4, height // 2)
        x, y = np.random.randint(0, width - w), np.random.randint(0, height - h)
        mask[x:x + w, y:y + h] = mask_color
    return mask

if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mask_color = (255, 0, 255)
    for i in range(args.count_videos):
        file_name = f"{args.out_dir}/{i}.gif"

        image_lst = [ create_random_mask(args.width, args.height, np.random.randint(1, 3))
                     for _ in range(args.frames_per_video)]
        imageio.mimsave(file_name, image_lst, fps=args.fps)