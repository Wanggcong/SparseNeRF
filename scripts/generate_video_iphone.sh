##########################################################################
# llff_rgbd
# scene01-scene08

root_path=iphone_ 
dataset=scene04
which_iter=60000 
postfix=debug0
video_name=rgb_depth # set --rgb_only True or False
python get_video.py --dataset $dataset --root_path $root_path  --postfix $postfix --video_name $video_name --which_iter $which_iter
# ##########################################################################

