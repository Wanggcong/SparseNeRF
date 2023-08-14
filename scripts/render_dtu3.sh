
# evaluation on 15 scenes:
# scan30,scan34,scan41,scan45, scan82,scan103, scan38, scan21
# scan40, scan55, scan63, scan31, scan8, scan110, scan114, 

CUDA_VISIBLE_DEVICES=0  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan30" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=1  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan34" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=2  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan41" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=3  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan45" --postfix "debug1" 

CUDA_VISIBLE_DEVICES=4  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan82" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=5  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan103" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=6  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan38" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=7  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan21" --postfix "debug1" 


CUDA_VISIBLE_DEVICES=0  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan40" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=1  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan55" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=2  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan63" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=3  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan31" --postfix "debug1" 

CUDA_VISIBLE_DEVICES=4  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan8" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=5  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan110" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=6  python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan114" --postfix "debug1" 
