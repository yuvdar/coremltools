# Copyright (c) 2017 Lightricks. All rights reserved.

from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

class ConditionalInstanceNormalization(Layer): # pylint: disable=too-many-instance-attributes
    """Conditional Instance normalization layer (Dumoulin et al., 2017).
    Contains 'num_of_style' different normalization of previous layer at each step.
    Each normalization is per channel mean and variance normalization and two trainable
    shift and scale.
    At each step one normalization is chosen (for either feed forward or back propagation)

    ConditionalInstanceNormalization is called with two layers - a trainable layer of arbitrary
    shape and a const layer of shape (1,1) indicating which normalization to choose

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `ConditionalInstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add a trainable offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, add a trainable multiply `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        First input:
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
        Second input: (1,1)
    # Output shape
        Same shape as first input.
    # Weights shape
        tuple - (number of normalization, previous layer channels)
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
        (https://arxiv.org/abs/1607.08022)
        - [A LEARNED REPRESENTATION FOR ARTISTIC STYLE](https://arxiv.org/abs/1610.07629)
    """
    def __init__(self, # pylint: disable=too-many-arguments
                 axis=None,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 num_of_styles=1,
                 **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.num_of_styles = num_of_styles
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        input_shape = input_shape[0]
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = [InputSpec(ndim=ndim), InputSpec(ndim=2)]
        if self.axis is None:
            shape = (1,)
        else:
            shape = (self.num_of_styles, input_shape[self.axis])

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        inputs, label = inputs
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [input_shape[0] or -1] + [1] * (len(input_shape) - 1)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
        if self.scale:
            broadcast_gamma = K.reshape(K.gather(self.gamma, label), broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(K.gather(self.beta, label), broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'num_of_styles' : self.num_of_styles,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConditionalInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if len(input_shape) == 4:
            output_shape = (input_shape[0], 1, 1, input_shape[self.axis])
        else:
            output_shape = (input_shape[0], input_shape[self.axis])
        return output_shape


get_custom_objects().update({'ConditionalInstanceNormalization': ConditionalInstanceNormalization})
