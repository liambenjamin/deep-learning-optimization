import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import architectures
from coadjoint import CoadjointModel



def generate_rnn_model(name, T, ft_dim, hid_dim, out_dim, penalty, penalty_weight, learning_rate, batch_size=32, invert_index=None, beta=0.0):
    """Generates coadjoint model from provided arguments

    Args:
        1. name (string): recurrent layer name
        2. T (int): time horizon of task
        3. ft_dim (int): feature dimension
        4. hid_dim (int): recurrent dimension
        5. out_dim (int): output (label) dimension
        6. penalty (tf.function): adjoint regularizer
        7. penalty_weight (float): penalty applied to adjoint regularizer
        8. learning_rate (float): learning rate used during training
        9. batch_size (int): batch size used for sgd routines (default=32)
        10. beta (float): optional parameter for tilted recurrent layer

    Returns: tensorflow keras model object with coadjoint training routine

    Note(s):
        1. Hard codes optimizer to Adam
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    output_activation = 'softmax' if out_dim > 1 else 'sigmoid'
    loss = tf.keras.losses.SparseCategoricalCrossentropy() if out_dim > 1 else tf.keras.losses.BinaryCrossentropy()

    layers = {'rnn': architectures.BasicRecurrentLayer(hid_dim, ft_dim, T),
              'tiltedrnn': architectures.TiltedRecurrentLayer(hid_dim, ft_dim, T, beta),
              'inversernn': architectures.InverseRecurrentLayer(hid_dim, ft_dim, T, invert_index),
              'lstm': architectures.LSTMRecurrentLayer(hid_dim),
              'gru': architectures.GRURecurrentLayer(hid_dim),
              'antisymmetric': architectures.AntisymmetricRecurrentLayer(hid_dim, ft_dim, epsilon=0.01, gamma=0.01, sigma=0.01),
              'temporalattention': architectures.TemporalAttentionLayer(hid_dim, T),
              'exponential': architectures.ExponentialRecurrentLayer(hid_dim),
              'lipschitz': architectures.LipschitzRecurrentLayer(hid_dim, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128),
              'unicornn': architectures.UnICORNNRecurrentLayer(hid_dim, ft_dim, epsilon=0.03, alpha=0.9, L=2)
             }
    
    inputs = tf.keras.Input(shape=(T,ft_dim), batch_size=batch_size, name='input-layer')
    rec_layer = layers[name]   
    dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer', trainable=True)
    
    states = rec_layer(inputs)
    outputs = dense_layer(states[-1])

    model = CoadjointModel(inputs=inputs, outputs=[outputs, states])
    model.compile(optimizer=optimizer,
            loss_fn=loss,
            lc_weights= [penalty_weight[0], penalty_weight[1]],
            adj_penalty=penalty,
            model_name=name,
            )
         
    return model