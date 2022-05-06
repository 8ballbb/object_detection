import tensorflow as tf
from .yolov4 import YOLO, decode, filter_boxes
from .utils import load_config, load_weights


def convert_darknet_to_tf(model_files, weights="yolov4-tiny_best.weights", input_size=416, score_threshold=.2, tiny=False, framework="tf"):
    strides, anchors, num_class, xyscale = load_config(f"{model_files}obj.names", tiny)
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLO(input_layer, num_class, tiny)
    bbox_tensors = []
    prob_tensors = []
    if tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 16, num_class, strides, anchors, i, xyscale, framework)
            else:
                output_tensors = decode(fm, input_size // 32, num_class, strides, anchors, i, xyscale, framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 8, num_class, strides, anchors, i, xyscale, framework)
            elif i == 1:
                output_tensors = decode(fm, input_size // 16, num_class, strides, anchors, i, xyscale, framework)
            else:
                output_tensors = decode(fm, input_size // 32, num_class, strides, anchors, i, xyscale, framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if framework == "tflite":
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(
            pred_bbox, pred_prob, score_threshold=score_threshold, 
            input_shape=tf.constant([input_size, input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    load_weights(model, f"{model_files}{weights}", tiny)
    model.summary()
    if framework == "tf":
        output = f"{model_files}{weights.replace('.weights', '_tf')}"
    elif framework == "tflite":
        output = f"{model_files}{weights.replace('.weights', '_tflite')}"
    else:
        output = f"{model_files}{weights.replace('.weights', '_trt')}"
    model.save(output)
