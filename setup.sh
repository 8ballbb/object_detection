# clone darknet GitHub repository
git clone https://github.com/AlexeyAB/darknet

# change makefile
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
sed -i 's/OPENMP=0/OPENMP=1/' Makefile

# build darknet
make
