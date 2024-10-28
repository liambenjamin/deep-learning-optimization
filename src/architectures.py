"""
File Description: Recurrent architecture (layer) implementations.

    Architectures:
        - Basic RNN
        - Tilted RNN
        - Inverse RNN
        - Antisymmetric RNN
        - Exponentail RNN
        - Lipschitz RNN
        - LSTM
        - GRU
        - UnICORNN
        - Attention (temporal)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

class BasicRecurrentLayer(tf.keras.layers.Layer):
    """Basic RNN Layer Class
    
    Args:
        1. units (int): recurrent dimension 
        2. ft_dim (int): input feature dimension 
        3. time_horizon (int): length of task time horizon 
    """

    def __init__(self, units, ft_dim, time_horizon):
        super(BasicRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        
    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
   
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R) 
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z) 
            states.append(state)
        
        return states  

    
class TiltedRecurrentLayer(tf.keras.layers.Layer):
    """Tilted RNN Layer
    
    Args:
        1. units (int): recurrent dimension 
        2. ft_dim (int): input feature dimension 
        3. time_horizon (int): length of task time horizon 
        4. beta (float): smoothing penalty weight
    """

    def __init__(self, units, ft_dim, time_horizon, beta):
        super(TiltedRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.beta = beta
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        
    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
   
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R) 
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z) 
            states.append(state)
        
        states_stack = tf.unstack(tf.stack(states, axis=1), axis=0) # B x [Txd]
        terminal_state = []
      
        for i in range(0,len(states_stack)):
            term1 = self.activation(tf.reduce_mean(tf.math.exp(self.beta * states_stack[i]), axis=0, keepdims=True))
            term2 = (1/self.beta) * tf.math.log(term1)
            terminal_state.append(term2)   
       
        terminal_state = tf.concat(terminal_state, axis=0) # (B,d)
        states.append(terminal_state)

        return states 


class InverseRecurrentLayer(tf.keras.layers.Layer):
    """Inverse RNN Layer class

    args:
        1. units (int): recurrent dimension 
        2. ft_dim (int): input feature dimension 
        3. time_horizon (int): length of task time horizon 
        4. invert_index (int in [0,T]): number of steps inbetween inverting W 
    """
    def __init__(self, units, ft_dim, time_horizon, invert_index):
        super(InverseRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.invert_index = invert_index
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def call(self, inputs):
        
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        W_inv = tf.linalg.inv(self.W)
        W = self.W
        ct=1

        for i in range(0,N):
            h = K.dot(input_seq[i], self.R)
            if i % self.invert_index == 0:
                ct = ct*-1
                W = self.W if ct > 0 else W_inv
    
            z = h + K.dot(state, W) + self.bias
            state = self.activation(z)
            states.append(state)

        return states


class AntisymmetricRecurrentLayer(tf.keras.layers.Layer):
    """Antisymmetric Recurrent Layer class
    
    args:
        1. units (int): recurrent dimension 
        2. ft_dim (int): input feature dimension 
        3. epsilon (float): discretization step size (euler method)
        4. gamma (float): diffusion parameter
        5. sigma (float): varianace of input-hidden matrix initialization
    """
    def __init__(self, units, ft_dim, epsilon=0.01, gamma=0.01, sigma=0.01):
        super(AntisymmetricRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.sigma = sigma
        self.ft_dim = ft_dim
        self.V = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.ft_dim, seed=0),
            trainable=True, name='V'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=self.sigma/self.units, seed=0),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')


    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        states = []

        for i in range(0,N):
            M = self.W - tf.transpose(self.W) - self.gamma * tf.eye(self.units)
            h = K.dot(input_seq[i], self.V)
            z = h + K.dot(state,M) + self.bias
            tanh_z = tf.keras.activations.tanh(z)
            state = state + self.epsilon * tanh_z
            states.append(state)

        return states


class LSTMRecurrentLayer(tf.keras.layers.Layer):
    """LSTM Recurrent Layer
    
    args:
        1. hid_dim: recurrent dimension
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.cell = tf.keras.layers.LSTMCell(hid_dim)
        self.hid_dim = hid_dim

    def build(self, input_shape):
        initial_states = self.cell.get_initial_state(batch_size=input_shape[0], dtype='float32')
        self.x0 = initial_states[0]
        self.c0 = initial_states[1]

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        state = self.x0
        cell_state = self.c0
        states = []
        cell_states = []

        for i in range(0,N):
            _, [state, cell_state] = self.cell(input_seq[i], [state, cell_state])
            states.append(state)
            cell_states.append(cell_state)

        return states #, cell_states

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'cell': self.cell,
            'hid_dim': self.hid_dim,
            'x0': self.x0,
            'c0': self.c0
        })
        return config


class GRURecurrentLayer(tf.keras.layers.Layer):
    """GRU Recurrent Layer
    
    args:
        1. hid_dim: recurrent dimension
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.cell = tf.keras.layers.GRUCell(hid_dim)
        self.hid_dim = hid_dim

    def build(self, input_shape):
        self.x0 = self.cell.get_initial_state(batch_size=input_shape[0], dtype='float32')

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        state = self.x0
        states = []

        for i in range(0,N):
            _, state = self.cell(input_seq[i], state)
            states.append(state)

        return states

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'cell': self.cell,
            'hid_dim': self.hid_dim,
            'x0': self.x0
        })
        return config

   
class TemporalAttentionLayer(tf.keras.layers.Layer):
    """Temporal input attention layer class (simple attention mechanism)

    args:
        1. units: attention dimension
        2. T: time horizon

    returns: context vector
    """ 
    def __init__(self, units, T):
        super(TemporalAttentionLayer,self).__init__()
        self.units = units
        self.T = T
        self.activation = tf.keras.activations.tanh
        self.W=self.add_weight(name='attention_weight', shape=(self.units,1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(self.T,1),
                               initializer='zeros', trainable=True)

    def call(self,inputs):

        inputs = tf.stack(inputs, axis=1) 
        h = self.activation(K.dot(inputs, self.W) + self.b) 
        alpha = tf.expand_dims(K.softmax(K.squeeze(h,axis=-1)), axis=-1)
        context = K.sum(inputs * alpha, axis=1) 
        
        return context
    

class ExponentialRecurrentLayer(tf.keras.layers.Layer):
    """Exponential RNN class
    
    args:
        1. units (int): recurrent dimension
    """

    def __init__(self, units):
        super(ExponentialRecurrentLayer, self).__init__()
        self.units = units

    def build(self, input_shape):

        self.T = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1/input_shape[-1]),
            trainable=True,
            name='T'
        )
        self.A = self.add_weight(
            shape=(self.units, self.units),
            initializer = CayleyInitialization,
            trainable=False,
            name='A'
        )
        self.B = tf.Variable(tf.linalg.expm(self.A), trainable=True)
        self.activation = modrelu(self.units)
        self.h0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def reset_parameters(self):
        # retraction to tangent space
        A = tf.linalg.band_part(self.A, 0, -1) # upper triangular matrix
        A = A - tf.transpose(A)
        self.A.assign(A)
        # assign B from retraction
        self.B.assign(tf.linalg.expm(self.A))

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        state = tf.ones(shape=(tf.shape(inputs)[0], self.units)) * self.h0
        states = []

        for i in range(0,N):
            h = K.dot(input_seq[i], self.T)
            h = h + K.dot(state, self.B)
            h = self.activation(h)
            state = h
            states.append(state)

        return states


def create_diag_(A, diag):
    """Cayley initialization helper"""

    n = A.shape[0]
    diag_z = np.zeros(n-1)
    diag_z[::2] = diag
    A_init = tf.linalg.diag(diag_z, k=1)
    A_init = A_init - tf.transpose(A_init)
    
    return A_init

def cayley_init_(A):
    """Cayley initialization helper"""
    
    size = A.shape[0] // 2
    diag = tf.random.uniform(shape=(size,), minval=0., maxval=np.pi / 2.)
    diag = -tf.sqrt( (1. - tf.cos(diag)) / (1. + tf.cos(diag)) )

    return create_diag_(A, diag)


class CayleyInitialization(tf.keras.initializers.Initializer):
    """Cayley initialization class for Exponential RNN"""

    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        A = tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        A_cayley = cayley_init_(A)
        return tf.cast(A_cayley, tf.float32)

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}


class modrelu(tf.keras.layers.Layer):
    """Modified ReLU activation function for Exponential RNN"""

    def __init__(self, dim):
        super(modrelu, self).__init__()
        self.dim = dim

    def build(self, inputs):
        self.bias = tf.Variable(tf.random.uniform(shape=(self.dim,), minval=-0.01, maxval=0.01), trainable=True, name='bias')

    def call(self, inputs):
        nrm = tf.abs(inputs)
        biased_nrm = nrm + self.bias
        magnitude = tf.keras.activations.relu(biased_nrm)
        phase = tf.sign(inputs)
        return phase * magnitude

    def get_config(self):
        base_config = super(modrelu, self).get_config()
        config = {'bias': self.bias}
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class LipschitzRecurrentLayer(tf.keras.layers.Layer):
    """Lipschitz RNN Class
    
    args:
        1. units (int): recurrent dimension
        2-6. beta -> sigma: architecture specific parameters
    """

    def __init__(self, units, beta=0.75, gamma_A=0.001, gamma_W=0.001, epsilon=0.03, sigma=0.1/128):
        super(LipschitzRecurrentLayer, self).__init__()
        self.units = units
        self.beta = beta
        self.gamma_A = gamma_A
        self.gamma_W = gamma_W
        self.epsilon = epsilon
        self.sigma = sigma
        self.M_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.sigma)
        self.U_init = tf.keras.initializers.GlorotUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.D_init = tf.keras.initializers.GlorotUniform()

    def build(self, input_shape):
        self.M_A = self.add_weight(shape=(self.units, self.units), initializer=self.M_init, trainable=True, name='M_A')
        self.M_W = self.add_weight(shape=(self.units, self.units), initializer=self.M_init, trainable=True, name='M_W')
        self.U = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.U_init, trainable=True, name='U')
        self.bias = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def compose_A(self):
        A = (1-self.beta) * (self.M_A + tf.transpose(self.M_A)) + self.beta * (self.M_A - tf.transpose(self.M_A))
        A = A - self.gamma_A * tf.eye(self.units)
        return A

    def compose_W(self):
        W = (1-self.beta) * (self.M_W + tf.transpose(self.M_W)) + self.beta * (self.M_W - tf.transpose(self.M_W))
        W = W - self.gamma_W * tf.eye(self.units)
        return W

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        A = self.compose_A()
        W = self.compose_W()

        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        states = []

        for i in range(0,N):
            h = K.dot(input_seq[i], self.U)
            z = h + K.dot(state,W) + self.bias
            tanh_z = tf.keras.activations.tanh(z)
            Ah = K.dot(state, A)
            state = state + self.epsilon * (Ah + tanh_z) 
            states.append(state)

        return states


class UnICORNNRecurrentLayer(tf.keras.layers.Layer):
    """ UniCORNN Class (assumes fixed layer depth=2)
    
    args:
        1. units (int): recurrent dimension
        2. ft_dim (float): feature dimension
        3. epsilon: architecture specific parameter
        4. alpha: architecture specific parameter
        5. L: architecture specific parameter (num of stacked layers)
    """

    def __init__(self, units, ft_dim, epsilon=0.03, alpha=0.9, L=2):
        super(UnICORNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.w_init = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
        self.V_init = tf.keras.initializers.HeUniform()
        self.bias_init = tf.keras.initializers.Zeros()
        self.c_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        #self.D_init = tf.keras.initializers.GlorotUniform()
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_init = tf.keras.initializers.Zeros()
        self.L = L
        self.rec_activation = tf.keras.activations.tanh
        self.time_activation = mod_tanh
        self.w1 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(self.units,), initializer=self.w_init, trainable=True, name='w2')
        self.V1 = self.add_weight(shape=(self.ft_dim, self.units), initializer=self.V_init, trainable=True, name='V1')
        self.V2 = self.add_weight(shape=(self.units, self.units), initializer=self.V_init, trainable=True, name='V2')
        self.b1 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='b1')
        self.b2 = self.add_weight(shape=(self.units, ), initializer=self.bias_init, trainable=True, name='b2')
        self.c1 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True, name='c1')
        self.c2 = self.add_weight(shape=(self.units, ), initializer=self.c_init, trainable=True, name='c2')

    def call(self, inputs):

        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)

        y1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        z1 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        y2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))
        z2 = tf.zeros(shape=(tf.shape(inputs)[0], self.units))

        states = []

        for i in range(0,N):

            # layer 1
            δ1 = self.epsilon * self.time_activation(self.c1)
            h = K.dot(input_seq[i], self.V1)
            h = self.rec_activation(h + tf.multiply(self.w1, y1) + self.b1)
            z1_nxt = z1 - δ1 * (h + self.alpha * y1)
            y1_nxt = y1 + δ1 * z1_nxt

            # layer 2
            δ2 = self.epsilon * self.time_activation(self.c2)
            h = K.dot(y1_nxt, self.V2)
            h = self.rec_activation(h + tf.multiply(self.w2, y2) + self.b2)
            z2_nxt = z2 - δ2 * (h + self.alpha * y2)
            y2_nxt = y2 + δ2 * z2_nxt

            # store states
            state_i = tf.concat([y1_nxt, z1_nxt, y2_nxt, z2_nxt], axis=1)
            states.append(state_i)

            # reset states
            y1 = state_i[:, :self.units] 
            z1 = state_i[:, self.units:2*self.units] 
            y2 = state_i[:, 2*self.units:3*self.units] 
            z2 = state_i[:, 3*self.units:] 

        return states


def mod_tanh(inputs):
    """Activation acting on time scale parameter (i.e. c_i)
        Used in UnICORNN architecture
    args:
        1. inputs (tensor)

    returns:
        1. Activated inputs
    """
    return 0.5 + 0.5 * tf.keras.activations.tanh(inputs/2)
