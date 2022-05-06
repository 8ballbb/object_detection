# Training a Custom Object Detection Model (YOLOv4)

The code and notebook in this repository has been developed specifically to run on Paperspace. However, it is likely with minimal modification, it should also work on other GPU instances. 

## Step 1 - Annotation of Custom Data for YOLOv4

To train custom YOLOv4 object detection we are required to have training image data in a specific format – each image should have a corresponding file that contains the coordinates of the objects present in the image.

There is no single standard format when it comes to image annotation i.e., COCO, Pascal VOC and YOLO formats. The YOLO format uses a `.txt` file with the same name is created for each image file in the same directory. Each file contains the annotations for the corresponding image file, that is object class, object coordinates, height and width.

```<object-class> <x_center> <y_center> <width> <height>```

where `<object-class>` is the ID of the object category, `<x_center>` and `<y_center>` are x and y coordinates of the center of the bounding box, and `<width>` and `<height>` are width and height of the bounding box. Coordinates are normalized with the width and height of the image.

For each object, a new line is created.

Below is an example of annotation in YOLO format where the image contains two different objects.

```
0 45 55 29 67
1 99 83 28 44
```

There are several tools to annotate images in this format. Both of the following tools allow you to label images and export labels in the required YOLO format.

### [LabelImg](https://github.com/tzutalin/labelImg)

This is an offline tool. Follow installation guide in [github page](https://github.com/tzutalin/labelImg).

### [Makesense.ai](https://www.makesense.ai/)

This is an online tool, which requires no setup.

## Step 2 - Setup Instance

Clone this repo on the Paperspace GPU instance and launch the `training.ipynb` kernel.

The first step after starting the `training.ipynb` notebook is to run the `setup.sh` script in the first cell. Note: `setup.sh` must have the executable permission set i.e., `chmod +x setup.sh`.

This script sets up the instance so that it can use Darknet to train a model.

- Clone Darknet Repo
- Edit Darknet Makefile
- Build Darknet
- Download YOLOv4-tiny weights
- Copy YOLOv4-tiny config

## Step 3 - Upload Data

Upload annotated files into the `data` folder i.e., all `jpg` images and corresponding `txt` annotation files.

## Step 4 - Setup Training

Split the labels into 3 sets (training, validation and test). The default split is 70% (training), 10% (validation) and 20% (test). These parameters can be changed.

The split is done in such a way that it applies at the class level and not the image level.

Three files containing a list of image filenames are created called `train.txt`, `val.txt` and `test.txt`. These are then located in a folder within `/content/object_detection/model_files/`.

Other files that are created that are required for training and evaluation:

* **obj.data and obj_test.data**

This file states how many classes there are, what the train and validation files are and which file contains the name of the object we want to detect. During training save the weight in the backup folder.

```
classes=3
train=train.txt
valid=val.txt
names=obj.names
backup=backup
```

```
classes=3
train=train.txt
valid=test.txt
names=obj.names
backup=backup
```

* **obj.names**

This is a file with the object label names in order.

```
car
truck
bus
```

### Edit config

There are many hyperparameters that can be changed in the config and it is possible to open that file and edit it. The following utility function is an aid to edit the primary hyperparameters without having to open the config file.

Possible hyperparameters to edit:
* height and width - both of these must be divisible by 32
* max_batch - minimum should be 2000 per class
* steps
* batch
* subdivisions - batch must be divisible by subdivisions

## Step 5 - Training

The next step is to train the model. The model will save checkpoints every 1000 epochs, the last weights and the best weights.

## Step 6 - Evaluation

The evaluation step uses the best weights and evaluates the model on the test set. It is possible to use other weights by changing the weights path.

## Step 7 - Convert Model

Models can be converted to multiple formats i.e., tensorflow, TFLite and TensorRT.
