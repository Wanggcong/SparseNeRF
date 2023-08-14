##########################################################################
# dtu
# scan
# 40, 55,63,110,114, 21, 30, 31,     8, 34, 41,45,82,103, 38

root_path=dtu3_sparsenerf_new4_
dataset=scan30
which_iter=90000
postfix=debug10
video_name=rgb_depth # set --rgb_only True or False
python get_video.py --dataset $dataset --root_path $root_path  --postfix $postfix --video_name $video_name --which_iter $which_iter
# ##########################################################################
