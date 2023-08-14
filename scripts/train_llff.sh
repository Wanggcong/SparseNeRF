
CUDA_VISIBLE_DEVICES=0  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fern" --postfix "debug0"
CUDA_VISIBLE_DEVICES=1  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "flower" --postfix "debug0"
CUDA_VISIBLE_DEVICES=2  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fortress" --postfix "debug0"
CUDA_VISIBLE_DEVICES=3  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "horns" --postfix "debug0"

CUDA_VISIBLE_DEVICES=4  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "leaves" --postfix "debug0"
CUDA_VISIBLE_DEVICES=5  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "orchids" --postfix "debug0"
CUDA_VISIBLE_DEVICES=6  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "room" --postfix "debug0"
CUDA_VISIBLE_DEVICES=7  python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "trex" --postfix "debug0"


