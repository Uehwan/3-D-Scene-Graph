cd ./FactorizableNet/lib
conda install pip pyyaml sympy h5py cython numpy scipy
conda install -c menpo opencv3
conda install -c soumith pytorch torchvision cuda80
pip install easydict
bash ./make.sh

cd ../../
touch sort/__init__.py
touch FactorizableNet/__init__.py
ln -s ./FactorizableNet/options/ options
mkdir data
ln -s ./FactorizableNet/data/svg data/svg
ln -s ./FactorizableNet/data/visual_genome data/visual_genome
ln -s ./sort/mot_benchmark data/mot_benchmark

