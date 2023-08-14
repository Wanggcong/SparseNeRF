

srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fern" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "flower" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fortress" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "horns" --postfix "debug0"

srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "leaves" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "orchids" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "room" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_llff_dtu.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "trex" --postfix "debug0"

srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fern" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "flower" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fortress" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "horns" --postfix "debug0"

srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "leaves" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "orchids" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "room" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python eval.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "trex" --postfix "debug0"


srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python render.py --gin_configs configs/llff3.gin --checkpoint_dir "llff3_"  --dataset_id "fern" --postfix "debug0"


srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python render.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_sparsenerf_new4_"  --dataset_id "scan30" --postfix "debug10"


srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python render.py --gin_configs configs/nvs_rgbd_kinect.gin --checkpoint_dir "llff3_sparsenerf_updated_new95_"  --dataset_id "scene01" --postfix "debug24"


srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python render.py --gin_configs configs/nvs_rgbd_zed.gin --checkpoint_dir "llff3_sparsenerf_updated_new97_"  --dataset_id "scene01" --postfix "debug12"




srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene01" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene02" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene03" --postfix "debug0"
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1 python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene04" --postfix "debug0"

srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=train_debug1 --kill-on-bad-exit=1   python train_iphone.py --gin_configs configs/nvs_rgbd_iphone.gin --checkpoint_dir "iphone_"  --dataset_id "scene01" --postfix "debug0"
