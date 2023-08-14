# nvs-rgbd, iphone

CUDA_VISIBLE_DEVICES=0  python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene01" --postfix "debug0"
CUDA_VISIBLE_DEVICES=1  python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene02" --postfix "debug0"
CUDA_VISIBLE_DEVICES=2  python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene03" --postfix "debug0"
CUDA_VISIBLE_DEVICES=3  python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene04" --postfix "debug0"


