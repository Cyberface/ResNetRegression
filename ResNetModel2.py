"""

Modifciation to the ResNet regression model by Chen,D+
Entropy 2020, 22, 193.

Here we:
    remove some of batch norm layers to see if
    that helps reach lower loss values

"""
from tensorflow.keras import layers, models


def identity_block(input_tensor, units):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
            input_tensor: input tensor
            units:output shape
    # Returns
            Output tensor for the block.
    """
    x = layers.Dense(units)(input_tensor)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(units)(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def dens_block(input_tensor, units):
    """A block that has a dense layer at shortcut.
    # Arguments
            input_tensor: input tensor
            unit: output tensor shape
    # Returns
            Output tensor for the block.
    """
    x = layers.Dense(units)(input_tensor)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(units)(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Dense(units)(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet2Regression(input_dim=1, output_dim=1, width=16, n_layers=2):
    """Instantiates the ResNet2 architecture.
    # Arguments
    input_dim {1}: X data will have shape, X.shape = (N, intput_dim)
    output_dim {1}: y data will have shape, y.shape = (N, output_dim)
            width {16}: width of each layer
            n_hidden_layers {2}: number of hidden ResNet blocks
    # Returns
            A Keras model instance.
    """
    Res_input = layers.Input(shape=(input_dim,))

    x = dens_block(Res_input, width)
    x = identity_block(x, width)
    x = identity_block(x, width)

    for i in range(n_layers):
        x = dens_block(x, width)
        x = identity_block(x, width)
        x = identity_block(x, width)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(inputs=Res_input, outputs=x)

    return model
