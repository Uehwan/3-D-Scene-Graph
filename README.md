git clone --recurse-submodules https://github.com/chickenbestlover/MSDN2

install pcl 1.7.2

sudo apt install python-sphinx doxygen doxygen-latex dh-exec build-essential devscripts

dget -u https://launchpad.net/ubuntu/+archive/primary/+files/pcl_1.7.2-14ubuntu1.16.04.1.dsc

cd pcl-1.7.2

sudo dpkg-buildpackage -r -uc -b

sudo dpkg -i pcl_*.deb

