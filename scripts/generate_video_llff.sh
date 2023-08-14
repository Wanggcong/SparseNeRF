##########################################################################
# llff3
# fern  flower  fortress  horns   leaves  orchids  room  trex
# --which_iter 69768
for dataset in  fern #fern  flower  fortress  horns   leaves  orchids  room  trex   
do
    root_path=llff3_
    dataset=$dataset
    postfix=debug0
    video_name=rgb_depth # set rgb_only true or false
    which_iter=90000
    python get_video.py --dataset $dataset --root_path $root_path  --postfix $postfix --video_name $video_name --which_iter $which_iter
done