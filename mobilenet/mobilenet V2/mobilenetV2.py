from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import h5py
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Add
from keras.layers import Dense
from keras.layers import DepthwiseConv2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications import imagenet_utils
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

# Load remote
BASE_WEIGHT_PATH = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/'


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
    function is different from `imagenet_utils.preprocess_input()`.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    x /= 128.
    x -= 1.
    return x.astype(np.float32)


def unprocess_input(x):
    """Unprocesses a numpy array encoding a batch of images.
    This function undoes the preprocessing which converts
    the RGB values from [0, 255] to [-1, 1].
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    x += 1.
    x *= 128.
    return x.astype(np.uint8)

# This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by 8
# It can be seen here  https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(input_shape=None,
                alpha=1.0,
                depth_multiplier=1,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000):
    """Instantiates the MobileNetV2 architecture.
    To load a MobileNetV2 model via `load_model`, import the custom
    objects `relu6` and pass them to the `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6})
    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape or invalid depth_multiplier, alpha,
            rows when weights='imagenet'
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')
    
    '''
    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = K.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = K.is_keras_tensor(
                    get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if K.image_data_format == 'channels_first':
                if input_tensor._keras_shape[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if input_tensor._keras_shape[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            K.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor,
                             'is type: ', type(input_tensor),
                             'which is not a valid type')

        if input_shape is None and not K.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and K.is_keras_tensor(input_tensor):
            if K.image_data_format() == 'channels_first':
                rows = input_tensor._keras_shape[2]
                cols = input_tensor._keras_shape[3]
            else:
                rows = input_tensor._keras_shape[1]
                cols = input_tensor._keras_shape[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224
   
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    '''
    input_shape=(224,224,3)
    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            if rows is None:
                rows = 224
                warnings.warn('MobileNet shape is undefined.'
                              ' Weights for input shape'
                              '(224, 224) will be loaded.')
            else:
                raise ValueError('If imagenet weights are being loaded, '
                                 'input must have a static square shape'
                                 '(one of (96, 96), (128, 128), (160, 160),'
                                 '(192, 192), or (224, 224)).'
                                 'Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv1')(img_input)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    x = _first_inverted_res_block(x,
                                  filters=16,
                                  alpha=alpha,
                                  stride=1,
                                  expansion=1,
                                  block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters,
               kernel_size=1,
               use_bias=False,
               name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax',
                  use_bias=True, name='Logits')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_first" format '
                             'are not available.')

        if include_top:
            model_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
                str(alpha) + '_' + str(rows) + '.h5'
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name, weigh_path,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
                str(alpha) + '_' + str(rows) + '_no_top' + '.h5'
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name, weigh_path,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Expand

    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
               use_bias=False, activation=None,
               name='mobl%d_conv_expand' % block_id)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_bn_expand' %
                           block_id)(x)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same',
                        name='mobl%d_conv_depthwise' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_depthwise' % block_id)(x)

    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name='mobl%d_conv_project' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_bn_project' % block_id)(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def _first_inverted_res_block(inputs,
                              expansion, stride,
                              alpha, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride, activation=None,
                        use_bias=False, padding='same',
                        name='mobl%d_conv_depthwise' %
                        block_id)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_depthwise' %
                           block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name='mobl%d_conv_project' %
               block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_project' %
                           block_id)(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


model=MobileNetV2(input_shape=(224,224,3))
print(model.summary())

