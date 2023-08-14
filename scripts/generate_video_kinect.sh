





##########################################################################
# nvs_rgbd kinect
# scene01-scene08
# 
root_path=llff3_sparsenerf_updated_new95_ 
dataset=scene01
which_iter=60000 # for lidar
postfix=debug24
video_name=rgb_depth # set --rgb_only True or False
python get_video.py --dataset $dataset --root_path $root_path  --postfix $postfix --video_name $video_name --which_iter $which_iter
# ##########################################################################




