###################################################################################
# #dataset_id: fern  flower  fortress  horns  leaves  orchids  room  trex
# benchmark=LLFF # LLFF
# dataset_id=fern
# root_path=/media/deep/HardDisk4T-new/datasets/nerf_llff_data-20220519T122018Z-001/nerf_llff_data/
# python get_depth_map.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id 


# dataset_id: fern  flower  fortress  horns  leaves  orchids  room  trex
benchmark=LLFF # LLFF
root_path=/mnt/lustre/gcwang/datasets/nerf_llff_data/

for dataset_id in  flower  fortress  horns  leaves  orchids  room  trex
do
    python get_depth_map_for_llff_dtu.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id 
done


