#! /bin/bash
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/arrayfire ..
make -j $(nproc)
sudo make install


examples/cuda/plot3
