# nvs-rgbd, Kinect

CUDA_VISIBLE_DEVICES=0  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene01" --postfix "debug0"
CUDA_VISIBLE_DEVICES=1  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene02" --postfix "debug0"
CUDA_VISIBLE_DEVICES=2  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene03" --postfix "debug0"
CUDA_VISIBLE_DEVICES=3  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene04" --postfix "debug0"
CUDA_VISIBLE_DEVICES=4  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene05" --postfix "debug0"
CUDA_VISIBLE_DEVICES=5  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene06" --postfix "debug0"
CUDA_VISIBLE_DEVICES=6  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene07" --postfix "debug0"
CUDA_VISIBLE_DEVICES=7  python train_kinect.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "kinect_"  --dataset_id "scene08" --postfix "debug0"



