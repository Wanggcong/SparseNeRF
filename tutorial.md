## How to apply SparseNeRF to your custom dataset.
* Create a datasets_depth_xyz.py (copy datasets_depth_llff_dtu.py) in ./internal
* Create a config in ./configs, say xyz.gin
* Set a new dataset_loader Config.dataset_loader = ‘xyz’ in xyz.gin
* Carefully set Config.near and Config.far in xyz.gin
In dataset_xyz.py, add a 

    dataset_dict = {
    ‘xyz': LLFF,
    'dtu': DTU,
    ‘’
    }
* Modify the Class LLFF: 
In _next_train_ function, define: box_h = 300 box_w = 400 
The box size depends on 1) the object size and 2)the accuracy of pre-trained depth estimation models. Please see https://github.com/Wanggcong/SparseNeRF/issues/7

    * Please not that the depth map of your code. In some depth maps, large values denote near (e.g., pre-trained depth estimation models output inverse depth maps). In this way, we use 
“if label0<=label1:” in Line 594.

    * In some cases, we might need to mask uncertain values, like Kinect.

* In train_xyz.py (copy train_llff.py) or render.py or eval.py, import the new file datasets_depth_xyz.py
* Add Config.dataset_loader==‘xyz’ option if any. (search ‘loader’ ) 
* Check if you need to adjust the weight of depth loss, including depth ranking loss and continuity loss. Please see https://github.com/Wanggcong/SparseNeRF/issues/8
* Check if you use the NDC coordinate or carefully adjust the loss weights to balance rgb color reconstruction and depth distillation losses. Please see the [link](https://github.com/Wanggcong/SparseNeRF/issues/6#issuecomment-1688433968)
