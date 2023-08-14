import cv2
import torch
import urllib.request

# import matplotlib.pyplot as plt
import utils.io

import numpy as np
import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--benchmark', type=str) 
parser.add_argument('-d', '--dataset_id', type=str)
parser.add_argument('-r', '--root_path', type=str)
args = parser.parse_args()



# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform



# scan8,scan21,30, 31,34,38,40,41,45,55,63,82,103,110,114,
if args.root_path[-1]!="/":
    root_path = args.root_path+'/'
else:
    root_path = args.root_path

# output_path = root_path
if args.benchmark=="DTU":
    root_path = root_path+args.dataset_id+'/*3_r5000*'
else:
    output_path = os.path.join(root_path+args.dataset_id, 'depth_maps')
    root_path = root_path+args.dataset_id+'/images_8/*png'
    # root_path = root_path+'/*png'

# output_path = os.path.join('depth_midas_temp/', args.benchmark, args.dataset_id)
output_path = os.path.join('depth_midas_temp_DPT_Hybrid/', args.benchmark, args.dataset_id)
# output_path = root_path #os.path.join('depth_midas_temp_MiDaS_small/', args.benchmark, args.dataset_id)

image_paths = sorted(glob.glob(root_path))
print('image_paths:', image_paths)
box_h = 384*2
box_w = 384*2

downsampling = 1
if not os.path.exists(output_path): 
    os.makedirs(output_path, exist_ok=True)
for k in range(len(image_paths)):
    filename = image_paths[k]
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('k, img.shape:', k, img.shape) #(1213, 1546, 3)
    h, w = img.shape[:2]
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    name = 'depth_'+filename.split('/')[-1]
    print('######### output_path and name:', output_path,  name)
    output_file_name = os.path.join(output_path, name.split('.')[0])
    # utils.io.write_depth(output_file_name.split('.')[0], output, bits=2)
    utils.io.write_depth(output_file_name, output, bits=2)