# 3D-Scene-Graph
3D scene graph generator from RGBD video, based on [FactorizableNet](https://github.com/yikang-li/FactorizableNet), implemented in Pytorch.

## Requirements
* Ubuntu 16.04
* Python 2.7
* Pytorch 0.3.1
* FactorizableNet
* TODO: write requirements in detail

## Installation
1. Download 3D-Scene-Graph repository 

```
    git clone --recurse-submodules https://github.com/Uehwan/3D-Scene-Graph.git
```
2. Install FactorizableNet
```
    cd 3D-Scene-Graph/FactorizableNet
```
Please follow the installation instructions in [FactorizableNet](https://github.com/yikang-li/FactorizableNet) repository, including downloading the pre-trained FactorizableNet trained from MSDN-DR-Net dataset.

3. Install 3D-Scene-Graph
```
   cd 3D-Scene-Graph
   ./build.sh
```



* 

* Install pcl
```
install pcl 1.7.2
sudo apt install python-sphinx doxygen doxygen-latex dh-exec build-essential devscripts
dget -u https://launchpad.net/ubuntu/+archive/primary/+files/pcl_1.7.2-14ubuntu1.16.04.1.dsc
cd pcl-1.7.2
sudo dpkg-buildpackage -r -uc -b
sudo dpkg -i pcl_*.deb
```

TODO: write installation in detail
TODO: add one-click installation script

## Example of usage

```
    python scene_graph_tuning.py --scannet_path data/lab_kitchen17/ --obj_thres 0.23 --thres_key 0.2 --thres_anchor 0.68 --visualize --frame_start 800 --plot_graph --disable_spurious --gain 10 --detect_cnt_thres 2 --triplet_thres 0.065
```

TODO: write examples of usage in detail

## Result

TODO: add result figures, plots, and 3d scene graphs.




