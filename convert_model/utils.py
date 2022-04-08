import numpy as np


def load_weights(model, weights_file, is_tiny=False):
    if is_tiny:
        layer_size = 21
        output_pos = [17, 20]
    else:
        layer_size = 110
        output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])
    wf.close()


def read_class_names(class_file_name):
    with open(class_file_name, "r") as f:
        class_names = [line.strip("\n") for line in f.readlines()]
    return class_names


def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def load_config(class_file_name, tiny):
    if tiny:
        strides = np.array([16, 32])
        anchors = get_anchors([23,27, 37,58, 81,82, 81,82, 135,169, 344,319], tiny)
        xyscale = [1.05, 1.05]
    else:
        strides = np.array([8, 16, 32])
        anchors = get_anchors([12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401], tiny)
        xyscale = [1.2, 1.1, 1.05]
    num_class = len(read_class_names(class_file_name))
    return strides, anchors, num_class, xyscale
