<div align="center">

<h1>SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis </h1>

<div>
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang</a>&emsp;
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a>&emsp;
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<div>
    Technical Report, 2023
</div>




### [Project](https://sparsenerf.github.io/) | [YouTube](https://www.youtube.com/watch?v=sHeWK1MSPg4) | [arXiv](https://arxiv.org/abs/2207.14811) 
<tr>
    <img src="https://github.com/wanggcong/wanggcong.github.io/blob/master/projects/SparseNeRF/static/images/demo_video_v4_scenes_only_sampled3.gif" width="90%"/>
</tr>
</div>

>**Abstract:** Neural Radiance Field (NeRF) significantly degrades when only a limited number of views are available. To complement the lack of 3D information, depth-based models, such as DSNeRF and MonoSDF, explicitly assume the availability of accurate depth maps of multiple views. They linearly scale the accurate depth maps as supervision to guide the predicted depth of few-shot NeRFs. However, accurate depth maps are difficult and expensive to capture due to wide-range depth distances in the wild. 

>In this work, we present a new Sparse-view NeRF (\textbf{SparseNeRF}) framework that exploits depth priors from real-world inaccurate observations. The inaccurate depth observations are either from pre-trained depth models or coarse depth maps of consumer-level depth sensors. Since coarse depth maps are not strictly scaled to the ground-truth depth maps, we propose a simple yet effective constraint, a local depth ranking method, on NeRFs such that the expected depth ranking of the NeRF is consistent with that of the coarse depth maps in local patches. To preserve the spatial continuity of the estimated depth of NeRF, we further propose a spatial continuity constraint to encourage the consistency of the expected depth continuity of NeRF with coarse depth maps. Surprisingly, with simple depth ranking constraints, SparseNeRF outperforms all state-of-the-art few-shot NeRF methods (including depth-based models) on standard LLFF and DTU datasets. Moreover, we collect a new dataset NVS-RGBD that contains real-world depth maps from Azure Kinect, ZED 2, and iPhone 13 Pro. Extensive experiments on NVS-RGBD dataset also validate the superiority and generalizability of SparseNeRF. Code and dataset will be released.
