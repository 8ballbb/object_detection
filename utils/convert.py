import tensorflow as tf
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils

# flags.DEFINE_string("weights", "./data/yolov4.weights", "path to weights file")

def main(
    weights, input_size=416, score_thres=.2, tiny=True, framework="tf", 
    output="/content/object_detection/model_files/yolov4_tiny"):
    """
    TODO: write docstring
    
    framework (tf, tflite)
    """
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(tiny)
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, tiny)
    bbox_tensors = []
    prob_tensors = []
    if tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            elif i == 1:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if framework == "tflite":
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(
            pred_bbox, pred_prob, 
            score_threshold=score_thres, 
            input_shape=tf.constant([input_size, input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(model, weights, tiny)
    model.summary()
    model.save(output)
