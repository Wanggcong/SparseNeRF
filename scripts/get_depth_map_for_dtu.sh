
###################################################################################
# get depth maps from large pre-trained models
# evaluation on 15 scenes:
# dataset_id: scan40, scan55, 63, 
# 110,114 21, 
# 30, 31, 8, #
# 34, 41,45,
# 82,103, 38

# benchmark=DTU #DTU 
# dataset_id=scan30
# root_path=/media/deep/HardDisk4T-new/datasets/DTU/Rectified
# python get_depth_map.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id
benchmark=DTU # LLFF
root_path=/mnt/lustre/gcwang/datasets/DTU/Rectified/

for dataset_id in  scan40  scan55  scan63  scan110  scan114  scan21  scan30 scan31 scan8 scan34 scan41 scan45 scan82 scan103 scan38
do
    python get_depth_map_for_llff_dtu.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id 
done





