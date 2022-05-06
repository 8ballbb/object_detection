# setup instance and opencv
apt-get update
apt-get upgrade -y
apt install libgl1-mesa-glx -y
apt-get install ffmpeg libsm6 libxext6  -y
apt install libopencv-dev -y
pip install opencv-python

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

# Fetch YOLOv4-tiny weights
cd ..
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29 -P $(pwd)/model_files/
# Copy Darknet config
cp $1darknet/cfg/yolov4-tiny-custom.cfg $1model_files/yolov4-tiny.cfg
