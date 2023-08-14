import cv2
import numpy as np
import glob

import imageio
import PIL.Image
import os

import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-r', '--root_path', type=str)
parser.add_argument('-p', '--postfix', type=str)
parser.add_argument('-n', '--video_name', type=str)
parser.add_argument('--which_iter', type=str, default="40000")

# Parse and print the results
args = parser.parse_args()

# fern  flower  fortress  horns  leaves  orchids  room  trex
# dataset = 'fortress'
root_path = f'./out/{args.root_path}{args.dataset}{args.postfix}/path_renders_step_{args.which_iter}/color_*png'
video_dir=f'./out/{args.root_path}{args.dataset}{args.postfix}/'

if not os.path.exists(video_dir):
  os.makedirs(video_dir)  
video_path=os.path.join(video_dir, f'{args.dataset}_{args.postfix}_{args.video_name}.mp4')

image_files = glob.glob(root_path)
image_files = sorted(image_files)
# print('############### image files:', image_files)

img_array = []
size = 0
for filename in image_files:
    target_pil = PIL.Image.open(filename).convert('RGB')
    target_pil = target_pil.resize((512, 384), PIL.Image.LANCZOS)
    img = np.array(target_pil, dtype=np.uint8)

    dirname = os.path.dirname(filename)
    colormap = 'depth_'+filename.split('/')[-1].split('_')[-1]
    depth_path = os.path.join(dirname, colormap)
    target_pil = PIL.Image.open(depth_path).convert('RGB')
    target_pil = target_pil.resize((512, 384), PIL.Image.LANCZOS)
    img_ = np.array(target_pil, dtype=np.uint8)    

    # img_array.append(img)
    img_array.append(np.concatenate([img, img_], axis=1))

video = imageio.get_writer(video_path, mode='I', fps=30, codec='libx264', bitrate='16M') 
for i in range(len(img_array)):
    video.append_data(img_array[i])
video.close()

