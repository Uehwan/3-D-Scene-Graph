# 3D-Scene-Graph: *A Sparse and Semantic Representation of Physical Environments for Intelligent Agents*
3D scene graph generator from RGBD video, based on [FactorizableNet](https://github.com/yikang-li/FactorizableNet), implemented in Pytorch.

<img src="vis_result/fig/3dsg_readme_fig.png" width="600">

## Requirements
* Ubuntu 16.04+
* Python 2.7
* Pytorch 0.3.1
* FactorizableNet
* TODO: write requirements in detail.

## Installation
1. Download 3D-Scene-Graph repository 

```
    git clone --recurse-submodules https://github.com/Uehwan/3D-Scene-Graph.git
```
2. Install FactorizableNet
```
    cd 3D-Scene-Graph/FactorizableNet
```
Please follow the installation instructions in [FactorizableNet](https://github.com/yikang-li/FactorizableNet) repository.
Follow steps 1 through 6. You can skip step 7. Download VG-DR-Net in step 8. You do not need to download other models.


3. Install 3D-Scene-Graph
```
   cd 3D-Scene-Graph
   ./build.sh
```

TODO: write installation in detail
TODO: add one-click installation script

## Example of usage

```
    python scene_graph_tuning.py --scannet_path data/lab_kitchen17/ --obj_thres 0.23 --thres_key 0.2 --thres_anchor 0.68 --visualize --frame_start 800 --plot_graph --disable_spurious --gain 10 --detect_cnt_thres 2 --triplet_thres 0.065
```

TODO: write examples of usage in detail

## Result

![scores1](./vis_result/fig/result1.png)

TODO: add more result figures, plots, and 3d scene graphs.



## Demo Video

[![Video Label](http://img.youtube.com/vi/DpW7eyF2HiI/0.jpg)](https://youtu.be/DpW7eyF2HiI)
