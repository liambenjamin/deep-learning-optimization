
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

"""
Custom gradient for tf.norm
    - L2 norm
"""
@tf.custom_gradient
def norm(x): #x (bs, hid_dim)
    ϵ = 1.0e-17
    nrm = tf.norm(x, axis=1, keepdims=True)
    def grad(dy):
        return dy * tf.math.divide(x,(nrm + ϵ))
    return nrm, grad

"""
Scaled variance adjoint penalty (L2 norm)
"""
@tf.function
def scaled_variance_adjoint_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    t1 = tf.zeros(nrms[0].shape)
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)

"""
Pascanu et. 2013 penalty
    - paper: "On the difficulty of training recurrent neural networks"
"""
@tf.function
def pascanu_adjoint_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    G = tf.zeros(nrms[0].shape)

    for i in range(0,N-1):
        G += (nrms[i] / nrms[i+1] - 1) ** 2

    return tf.reduce_mean(G)

@tf.function
def forward_backward_adjoint_penalty(adjoints):

    N = len(adjoints)
    adjoints_shuffle =  [adjoints[i] for i in np.random.permutation(N)]
    nrms = [norm(adjoints[i]) for i in range(0,N)]
    nrms_2 = [norm(adjoints_shuffle[i]) for i in range(0,N)]

    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - nrms_2[i]) ** 2

    return tf.reduce_mean(G)

@tf.function
def normalized_adjoint_penalty(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]
    adjoints = tf.stack(adjoints, axis=1)
    adjoints_nrms = tf.unstack(adjoints / tf.linalg.norm(adjoints, axis=1, keepdims=True), axis=1)

    normalized_nrms = tf.stack([norm(adjoints_nrms[i]) for i in range(0,N)],axis=1)
    G = tf.math.reduce_variance(normalized_nrms, axis=0)

    return tf.reduce_mean(G)

"""
Terminal variance adjoint penalty (L2 norm)
"""
@tf.function
def terminal_adjoint_variance_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - nrms[-1]) ** 2

    return tf.reduce_mean(G)

"""
Non-Terminal variance adjoint penalty (L2 norm)
"""
@tf.function
def non_terminal_adjoint_variance_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    G_initial = (nrms[0] - nrms[-1]) ** 2

    nrms_matrix = tf.stack(adjoints[0:len(adjoints)-1], axis=1)
    nrms_matrix = nrms_matrix @ tf.transpose(nrms_matrix, [0,2,1])
    off_diag_matrix = nrms_matrix - tf.linalg.band_part(nrms_matrix, 0, 0)

    #nrms_matrix = tf.stack(nrms[0:len(nrms)-1], axis=1)
    G = tf.math.reduce_mean(off_diag_matrix)

    return tf.reduce_mean(G) + tf.reduce_mean(G_initial)

"""
input-output adjoint variance penalty (L2 norm)
"""
@tf.function
def input_output_adjoint_norm_variance_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]
    G = (nrms[0] - nrms[-1]) ** 2 
    
    return tf.reduce_mean(G)


"""
output jacobian variance adjoint penalty (L2 norm)
"""
@tf.function
def output_jacobian_norm_variance_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]

    t1 = tf.zeros(nrms[0].shape)
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)


@tf.function
def test_variance_penalty_L2(adjoints):

    N = len(adjoints)
    a_nrms = [norm(adjoints[i]) for i in range(0,N)]
    
    G = tf.zeros(a_nrms[0].shape)

    for i in range(0,N):
        G += a_nrms[i]

    
    G2 = tf.zeros(a_nrms[0].shape)

    for i in range(0,N):
        G2 += (a_nrms[i] / G - 1/N)**2

    
    return tf.reduce_mean(G2) 

"""
input-output jacobian variance adjoint penalty (L2 norm)
"""
@tf.function
def input_output_jacobian_norm_variance_penalty_L2(adjoints):

    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]
    G = (nrms[0] - nrms[-1]) ** 2 

    return tf.reduce_mean(G)

"""
input-output-middle jacobian variance adjoint penalty (L2 norm)
"""
@tf.function
def jacobian_variance_penalty_L2(adjoints):

    adjoints = tf.stack(adjoints, axis=1)
    A = tf.reduce_mean(adjoints, axis=0)
    G = A @ tf.transpose(A)

    return tf.math.reduce_variance(G)


"""
Scaled variance adjoint penalty (L1 norm)
"""
@tf.function
def scaled_variance_adjoint_penalty_L1(adjoints):
    
    N = len(adjoints)
    nrms = [tf.norm(adjoints[i], axis=1, ord=1) for i in range(0,N)]

    t1 = tf.zeros(nrms[0].shape)
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    G = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)

"""
Basic RNN class
"""
class RNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(RNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
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
    
"""
Component RNN class
"""
class ComponentRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(ComponentRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R_list = [self.add_weight(shape=(self.ft_dim, self.units),
                                       initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
                                       trainable=True, name='R_{0}'.format(i)) for i in range(0,self.time_horizon)
                                       ]
        self.W_list = [self.add_weight(shape=(self.units, self.units), 
                                       initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
                                       trainable=True, name = 'W_{0}'.format(i)) for i in range(0,self.time_horizon)
                                       ]
        self.bias_list = [self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias_{0}'.format(i)) for i in range(0,self.time_horizon)]
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        
    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
   
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R_list[i]) 
            z = h + K.dot(state, self.W_list[i]) + self.bias_list[i]
            state = self.activation(z) 
            states.append(state)
        
        return states  
    

"""
Test RNN class
"""
class TestRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(TestRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.orthogonal_init = tf.compat.v1.orthogonal_initializer()
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
            state = self.activation(z) * (tf.linalg.norm(self.W) * 0.5)
            states.append(state)

        #terminal_state = states[0]
        #for i in range(1,len(states)):
        #    terminal_state += states[i]
        #states.append(terminal_state / N)
        
        return states  
    
"""
Tilted RNN class
"""
class TiltedRecurrentLayer(tf.keras.layers.Layer):
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
            term1 = self.activation(tf.reduce_mean(tf.math.exp(self.beta * states_stack[i]), axis=0, keepdims=True)) # (1,4)
            term2 = (1/self.beta) * tf.math.log(term1)
            terminal_state.append(term2)   
       
        terminal_state = tf.concat(terminal_state, axis=0) # (B,d)
        #terminal_state = terminal_state * (tf.linalg.norm(states[-1], axis=-1,keepdims=True)/tf.linalg.norm(terminal_state, axis=-1,keepdims=True))
        states.append(terminal_state)

        return states 

"""
Controller Recurrent Layer
"""
class LinearRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(LinearRecurrentLayer, self).__init__()
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
        self.W_attn = self.add_weight(
            shape=(self.units,),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='W_attn'
        )
        self.bias_attn_list = [self.add_weight(shape=(4,), initializer="zeros", trainable=True, name='bias_attn_{0}'.format(i)) for i in range(0,self.time_horizon)]
        
    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
   
        for i in range(0,N):
            
            h = K.dot(input_seq[i], self.R)
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z) 
           
            #print(f'Wx: {Wx.shape}')
            state = self.activation(state * self.W_attn + self.bias_attn_list[i])
            #print(f'state: {state.shape}')
            states.append(state)

        return states

"""
Monic RNN class
"""
class MonicRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(MonicRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        self.a_seq = [self.add_weight(shape=(1,), initializer="random_normal", trainable=True, name='a_{0}'.format(i)) for i in range(0,self.time_horizon)]
        
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

        terminal_state = sum([self.a_seq[i] * states[i] for i in range(0,len(states))]) + (self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units)))
        #terminal_state = sum(states) + (self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units)))
        states.append(terminal_state)
        return states  


"""
Basic RNN class
"""
class NormRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(NormRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
    
    def unit_sphere(self, state, index):
        base = tf.ones(state.shape) * index / self.time_horizon
        center = state - base
        nrm = tf.linalg.norm(center, axis=-1, keepdims=True)
        radius = (1 / nrm) * center
        sphere = radius + base
        return sphere

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        #terminal_state = tf.zeros(state.shape)
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R)        
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z)
            #state = self.unit_sphere(state, i+1)
            states.append(state)
            #terminal_state += state_sphere
        
        states.append(state - self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units)))
        return states


"""
Basic RNN class
"""
class NewRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(NewRNNRecurrentLayer, self).__init__()
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

        neg_states = []
        neg_state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))

        #diff_state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        #out_states = []
   
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R)
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z)
            states.append(state)
            
            h = K.dot(input_seq[i]*-1, self.R)
            z = h + K.dot(neg_state, self.W) + self.bias
            neg_state = self.activation(z)
            neg_states.append(neg_state)

            #diff_state = self.activation(K.dot(state - neg_state, self.W2))# + diff_state) #K.dot(diff_state, self.R2)
            #diff_state = self.activation(diff_state)
            #out_states.append(diff_state)
            #state = diff_state
            #neg_state = diff_state
        #out_states.append(sum(out_states))
        #out = sum(out_states)
        #out_states.append(out)
        #terminal_state = state - neg_state #tf.concat((state,neg_state), axis=-1)
        #states.append(terminal_state)
        out_state = state - neg_state
        return (out_state, states, neg_states) #(states, neg_states)


"""
Filtering RNN class
"""
class FilteringRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(FilteringRNNRecurrentLayer, self).__init__()
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
            state = self.activation(z) + state    
            states.append(state)
        
        return states

"""
Input Attention RNN class
"""
class InputAttentionRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(InputAttentionRNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        #self.num_splits = num_splits
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        self.W_attn = self.add_weight(name='attention_weight', shape=(self.units,1), initializer='random_normal', trainable=True)
        self.bias_attn = self.add_weight(name='attention_bias', shape=(self.time_horizon,1), initializer='zeros', trainable=True)
        
    def compute_temporal_index_weighting(self, states):
        states = tf.stack(states, axis=1)
        # Alignment scores. Pass them through tanh function
        h = K.dot(states, self.W_attn) + self.bias_attn #[bs,T,1]
        h = self.activation(h) # [bs,T,1]
        alpha = K.softmax(K.squeeze(h,axis=-1)) # [bs, T]
        alpha = tf.unstack(K.expand_dims(alpha, axis=-1),axis=1) # T x [bs,1]

        return alpha #alpha_mask

        
    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        original_states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
     
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R)         
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z)
            original_states.append(state)

        alpha = self.compute_temporal_index_weighting(original_states)
        filtered_states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
     
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R) * alpha[i]         
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z)
            filtered_states.append(state)

        return (filtered_states, original_states, alpha)  
    
"""
Threaded Attention class
"""
class ThreadLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(ThreadLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R_thread = self.add_weight(
            shape=(self.time_horizon, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R_thread'
        )
        self.bias_R = self.add_weight(shape=(1,self.units), initializer="zeros", trainable=True, name='bias_R')
        self.W_attn_R = self.add_weight(name='attention_weight_R', shape=(self.ft_dim,1), initializer='random_normal', trainable=True)
        self.bias_attn_R = self.add_weight(name='attention_bias_R', shape=(self.units,1), initializer='zeros', trainable=True)
        
       
    def compute_context(self, states_in, attn_matrix, attn_bias):
        q = self.activation(K.dot(states_in, attn_matrix) + attn_bias) #[bs,d,1]
        alpha = K.expand_dims(K.softmax(K.squeeze(q,axis=-1)), axis=-1) #[bs,d,1]
        state = states_in * alpha # (bs,d,1)
        return (state, alpha) # (bs,d,p) & (bs,d,p)

    def call(self, inputs): # (B, T, 1)

        # thread rnn
        inputs_transpose = tf.transpose(inputs, [0,2,1])
        hR = tf.transpose(self.activation(K.dot(inputs_transpose, self.R_thread) + self.bias_R), [0,2,1])
        (hR_state, alpha_R) = self.compute_context(hR, self.W_attn_R, self.bias_attn_R)
        state_out = tf.reduce_sum(hR_state, axis=-1)
        
        return state_out

"""
Basic RNN class
"""
class ReservoirLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(ReservoirLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=False, name='R'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=False, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=False, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=False, name='initial_state')

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



"""
Basic RNN class
"""
class TransportRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(TransportRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        self.Q = self.add_weight(
            shape=(2, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='Q'
        )

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        mu = tf.reduce_mean(state, axis=-1, keepdims=True)
        sigma = tf.math.reduce_variance(state, axis=-1, keepdims=True)
        stats = tf.concat((mu,sigma),axis=-1)
     
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R) + K.dot(stats, self.Q)     
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z)
            states.append(state)
            mu = tf.reduce_mean(state, axis=-1, keepdims=True)
            sigma = tf.math.reduce_variance(state, axis=-1, keepdims=True)
            stats = tf.concat((mu,sigma),axis=-1)
        
        return states

"""
Dynamical RNN class
"""
class DynamicalRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(DynamicalRNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.complete_list = []
        self.new_indices = []
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.Q = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'Q'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))

        states_end = [tf.zeros(shape=(1,4)) for i in range(0,inputs.shape[0])]
    
        for i in range(0,N):
            h = K.dot(input_seq[i], self.R)         
            z = h + K.dot(state, self.W) + self.bias
            state = self.activation(z)
            states.append(state)

            #knowledge = tf.reduce_sum(self.activation(K.dot(state, self.Q)),axis=-1)
            #condition = tf.math.greater(knowledge, 1)

            if i > 2:
                state_unstack = tf.unstack(state, axis=0)
                state_unstack = [tf.expand_dims(j,axis=0) for j in state_unstack]
                tensor_condition(self, state_unstack)
            
                for j in range(0,len(self.new_indices)):
                    if j not in self.complete_list:
                        index = self.new_indices[j]
                        print(state_unstack[index])
                        states_end[index] = state_unstack[index]
                        self.complete_list.append(index)
                        
            self.new_indices = []
            #for j in range(0,len(state_unstack)):
                #if j not in complete_list:
            #    state_val = tf.cond(tf.reduce_sum(state_unstack[i]) > 1, lambda: tf.expand_dims(state_unstack,axis=0), lambda: tf.zeros(shape=(1,self.units)))
            #    states_end[i] += state_val
                                    

        states_end = tf.concat(states_end, axis=0)
        states.append(states_end)
        return states 

@tf.function
def tensor_condition(model, state_list):
  
    state_append = []
    for i in range(0,len(state_list)):
        if i not in model.complete_list:
            if tf.reduce_sum(state_list[i]) > 1:
                model.new_indices.append(i)
                #list_append.append(i)
                state_append.append(state_list[i])
    
    return #state_append



"""
Fuse RNN class
"""
class FuseRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(FuseRNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.linear_activation = tf.keras.activations.linear
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

        self.R2 = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R2'
        )
        self.W2 = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W2'
        )
        self.bias2 = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias2')
        self.R3 = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R3'
        )
        self.W3 = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(gain=1.25,seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W3'
        )
        self.bias3 = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias3')
        self.R4 = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='R4'
        )
        self.W4 = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(gain=0.75,seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W4'
        )
        self.bias4 = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias4')
        

        self.W_fuse =self.add_weight(name='fuse_weight', shape=(self.units,1),
                               initializer='random_normal', trainable=True)
        self.b_fuse =self.add_weight(name='fuse_bias', shape=(4,1),
                               initializer='zeros', trainable=True)

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        
        for i in range(0,N):
            
            h = K.dot(input_seq[i], self.R)
            z = h + K.dot(state, self.W) + self.bias
            state1 = self.linear_activation(z)
            #states.append(state)

            h = K.dot(input_seq[i], self.R2)
            z = h + K.dot(state, self.W2) + self.bias2
            state2 = self.linear_activation(z)
            #states2.append(state2)

            h = K.dot(input_seq[i], self.R3)
            z = h + K.dot(state, self.W3) + self.bias3
            state3 = self.linear_activation(z)

            h = K.dot(input_seq[i], self.R4)
            z = h + K.dot(state, self.W4) + self.bias4
            state4 = self.linear_activation(z)

            input_merge = tf.stack((state1, state2, state3, state4), axis=1)
            h = K.dot(input_merge, self.W_fuse) + self.b_fuse #[bs,T,1]
            h = self.activation(h) # [bs,T,1]
            alpha = K.softmax(K.squeeze(h,axis=-1))
            alpha = K.expand_dims(alpha, axis=-1) # [bs,T,1]
            context = input_merge * alpha # [bs ,100, 4]
            state = K.sum(context, axis=1) #[bs, 4]
            states.append(state) #(bs,d)

        return states
       

"""
Shielded RNN class
"""
class ShieldedRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(ShieldedRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')
        self.q = self.add_weight(
            shape=(self.units,1),
            initializer = tf.keras.initializers.RandomNormal(),
            trainable=True, name='q'
        )
        self.bias_beta = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias_beta_')

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        
        for i in range(0,N):
            
            Ru = K.dot(input_seq[i], self.R)
            Wx_b = K.dot(state, self.W) + self.bias
            x1 = self.activation(Ru + Wx_b)
            x2 = self.activation(Wx_b)

            qx1 = K.dot(x1, self.q) + self.bias_beta
            qx2 = K.dot(x2, self.q) + self.bias_beta
            Qx = tf.concat((qx1,qx2), axis=-1)
            Qx_sm = K.softmax(Qx, axis=-1)
            [beta_1, beta_2] = tf.unstack(Qx_sm, axis=-1)

            state = tf.expand_dims(beta_1,axis=-1) * x1 + tf.expand_dims(beta_2,axis=-1) * x2
            states.append(state)

        return states    


"""
Double RNN class
"""
class DoubleRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(DoubleRNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.units, seed=self.seed1),
            trainable=True, name='R'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.Q = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2+1),
            trainable=True, name = 'Q'
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
            if i > 0:
                z = z + 0.5 * (z - K.dot(states[-1], self.Q)) ** 2
            state = self.activation(z)
            states.append(state)

        return states    

"""
Basic RNN class
"""
class LinearStatesRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(LinearStatesRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        linear_states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        
        for i in range(0,N):
            
            h = K.dot(input_seq[i], self.R)
            z = h + K.dot(state, self.W) + self.bias
            linear_states.append(z)
            state = self.activation(z)
            states.append(state)

        return (states, linear_states)   



"""
Constrained RNN class
"""
class ConstrainedRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(ConstrainedRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
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
            if i == 0:
                h = K.dot(input_seq[i], self.R)
                z = h + K.dot(state, self.W) + self.bias
                state = self.activation(z)
                states.append(state)
            else:
                h = K.dot(input_seq[i], self.R)
                z = h + K.dot(state, self.W) + self.bias
                state = self.activation(z) - sum(states)
                states.append(state)

        return states  

"""
Basic RNN class
"""
class CompressionRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(CompressionRNNRecurrentLayer, self).__init__()
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
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W'
        )
        self.W_compression = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
            trainable=True, name = 'W_compression'
        )
        self.R_compression = self.add_weight(
            shape=(self.units * 10, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon*2, seed=self.seed1),
            trainable=True, name='R_compression'
        )
        self.bias_compression = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias_compression')
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

        # states: len = N & dim = (32, d*5)
        input_states = [tf.concat(states[i:i+10], axis=-1) for i in range(0,len(states),10)] # (32, d) #time
        compression_states = []
        c_state = tf.zeros(shape=state.shape) # (32,d)
        for i in range(1,len(input_states)):
            h = K.dot(input_states[i], self.R_compression)
            z = h + K.dot(c_state, self.W_compression) + self.bias_compression
            c_state = self.activation(z)
            compression_states.append(c_state)
        
        return (states, compression_states)

"""
Fixed Input-to-Hidden Matrix (R) RNN class
"""
class FixedInputRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(FixedInputRNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.seed1 = int(np.random.permutation(7555)[0])
        self.seed2 = int(np.random.permutation(7555)[0])
        self.R = self.add_weight(
            shape=(self.ft_dim, self.units),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=False, name='R'
        )
        self.W = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(seed=self.seed2),
            #regularizer=tf.keras.regularizers.L2(0.01),
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

"""
Fixed Hidden-to-Hidden Matrix (W) RNN class
"""
class FixedRecurrentRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(FixedRecurrentRNNRecurrentLayer, self).__init__()
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
            trainable=False, name = 'W'
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

"""
Inverse Penalty RNN class
"""
class InversePenaltyRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon, invert_index):
        super(InversePenaltyRNNRecurrentLayer, self).__init__()
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
        self.W1 = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=self.seed2),
            trainable=True, name = 'W1'
        )
        self.W2 = self.add_weight(
            shape=(self.units, self.units),
            initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=self.seed1),
            trainable=True, name = 'W2'
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')


    def call(self, inputs):
        
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        W = self.W1
        ct=1

        for i in range(0,N):
            
            h = K.dot(input_seq[i], self.R)

            if i % self.invert_index == 0:
                ct *= -1
                W = self.W1 if ct > 0 else self.W2
    
            z = h + K.dot(state, W) + self.bias

            state = self.activation(z)
            states.append(state)

        return states    

"""
Inverse RNN class
"""
class ExactInverseRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon, invert_index):
        super(ExactInverseRNNRecurrentLayer, self).__init__()
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

class LUInitialization(tf.keras.initializers.Initializer):
    def __init__(self, return_value):
        self.return_value = return_value
        self.initializer = tf.keras.initializers.Orthogonal()

    def __call__(self, shape, dtype=None, **kwargs):

        tf.keras.utils.set_random_seed(0)
        M = self.initializer(shape=(shape[0],shape[0])) if len(shape) == 1 else self.initializer(shape=shape)
        LU_M, p_M = tf.linalg.lu(M)
        W = tf.cast(M.numpy()[p_M], dtype=tf.dtypes.float32)
        LU, p = tf.linalg.lu(W)

        if self.return_value == 'matrix':
            return LU
        elif self.return_value == 'permutation':
            return p

"""
LU RNN class
"""
class LURNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon, invert_index):
        super(LURNNRecurrentLayer, self).__init__()
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
        self.LU = self.add_weight(
            shape=(self.units, self.units),
            initializer = LUInitialization(return_value='matrix'),
            trainable=True, name = 'LU'
        )
        self.p = self.add_weight(
            shape=(self.units,),
            initializer = LUInitialization(return_value='permutation'),
            trainable=False, name = 'p', dtype=tf.dtypes.int32
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name='bias')
        self.x0 = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Zeros(), trainable=True, name='initial_state')

    def construct_W(self):
        W = tf.linalg.lu_reconstruct(self.LU, self.p)
        return W
    
    def construct_W_inverse(self):
        W = tf.linalg.lu_matrix_inverse(self.LU, self.p)
        return W

    def call(self, inputs):
        
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        W_inv = self.construct_W_inverse()
        W_orig = self.construct_W()
        W = W_orig

        ct=1
        
        for i in range(0,N):
            
            h = K.dot(input_seq[i], self.R)

            if i % self.invert_index == 0:
                ct = ct*-1
                W = W_orig if ct < 0 else W_inv
    
            z = h + K.dot(state, W) + self.bias

            state = self.activation(z)
            states.append(state)

        return states    


"""
Covariance RNN class
"""
class CovarianceRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(CovarianceRNNRecurrentLayer, self).__init__()
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
        self.W_states = self.add_weight(
            shape=(self.time_horizon, 1),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='W_states'
        )
        self.W_linear_states = self.add_weight(
            shape=(self.time_horizon, 1),
            initializer = tf.keras.initializers.RandomNormal(stddev=1/self.time_horizon, seed=self.seed1),
            trainable=True, name='W_linear_states'
        )

    def call(self, inputs):
        input_seq = tf.unstack(inputs, axis=1)
        N = len(input_seq)
        states = []
        linear_states = []
        state = self.x0 * tf.ones(shape=(tf.shape(inputs)[0], self.units))
        
        for i in range(0,N):
            
            h = K.dot(input_seq[i], self.R)
            z = h + K.dot(state, self.W) + self.bias
            linear_states.append(z)
            state = self.activation(z)
            states.append(state)

        stacked_states = tf.transpose(tf.stack(states, axis=1), [0,2,1])
        states_out = self.activation(tf.squeeze(K.dot(stacked_states, self.W_states), axis=-1))

        stacked_linear_states = tf.transpose(tf.stack(linear_states, axis=1), [0,2,1])
        linear_states_out = self.activation(tf.squeeze(K.dot(stacked_linear_states, self.W_linear_states), axis=-1))

        states.append(states_out * linear_states_out)

        return states

"""
Basic RNN class
"""
class TransposeRNNRecurrentLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, time_horizon):
        super(TransposeRNNRecurrentLayer, self).__init__()
        self.units = units
        self.ft_dim = ft_dim
        self.time_horizon = time_horizon
        self.activation = tf.keras.activations.tanh
        self.linear_activation = tf.keras.activations.linear
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
            #regularizer=tf.keras.regularizers.L2(0.01),
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


        xT = tf.expand_dims(states[-1], axis=-1)
        xxT = xT @ tf.transpose(xT,[0,2,1])
        
        m_state = xxT @ xT
        m_end = m_state @ tf.transpose(m_state,[0,2,1])
        terminal_state = self.linear_activation(tf.squeeze(m_end @ m_state, axis=-1))
        states.append(terminal_state)
        #states.append(tf.squeeze(m_end @ m_state, axis=-1))

        return states

"""
Antisymmetric RNN class
"""
class antiRNNLayer(tf.keras.layers.Layer):
    def __init__(self, units, ft_dim, epsilon=0.01, gamma=0.01, sigma=0.01):
        super(antiRNNLayer, self).__init__()
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

"""
Description: LSTM layer class
"""
class LSTMLayer(tf.keras.layers.Layer):
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


"""
Description: GRU layer class
"""
class GRULayer(tf.keras.layers.Layer):
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

"""
temporal input attention layer class (simple attention mechanism)
"""    
class TemporalAttentionLayer(tf.keras.layers.Layer):
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

        #print('inputs:', inputs.shape)
        inputs = tf.stack(inputs, axis=1) #[bs,T,d]
        # Alignment scores. Pass them through tanh function
        h = self.activation(K.dot(inputs, self.W) + self.b) #[bs,T,1]
        alpha = K.softmax(K.squeeze(h,axis=-1)) #[bs,T]
        anti_alpha = tf.ones(alpha.shape) - alpha # (bs,T)
        anti_alpha = tf.expand_dims(anti_alpha / tf.reduce_sum(anti_alpha,axis=1,keepdims=True),axis=-1)
        alpha = tf.expand_dims(alpha, axis=-1) #[bs,T,1]
        context = K.sum(inputs * alpha, axis=1) #[bs, 4]
        anti_context = K.sum(inputs * anti_alpha, axis=1)
      
        return (context, anti_context)

"""
width attention layer class (simple attention mechanism)
"""
class WidthAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,units, T):
        super(WidthAttentionLayer,self).__init__()
        self.units = units
        self.T = T
        self.activation = tf.keras.activations.tanh
        self.W=self.add_weight(name='attention_weight', shape=(self.T,1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(self.units,1),
                               initializer='zeros', trainable=True)

    def call(self,inputs):
        
        inputs = tf.stack(inputs, axis=1) #[bs,T,d]
        inputs = tf.transpose(inputs, [0,2,1]) #[bs,d,T]
        # Alignment scores. Pass them through tanh function
        h = K.dot(inputs, self.W) + self.b #[bs,d,1]
        h = self.activation(h) # [bs,d,1]
        alpha = K.softmax(K.squeeze(h,axis=-1))
        alpha = K.expand_dims(alpha, axis=-1) # [bs,d,1]
        context = inputs * alpha # [bs ,d, T]
        context = tf.transpose(context, [0,2,1]) #[bs,T,d]
        context = K.sum(context, axis=1) #[bs, 4]
      
        return context

"""
(width + temporal) attention layer class (simple attention mechanism)
"""
class WidthTemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,units, T):
        super(WidthTemporalAttentionLayer,self).__init__()
        self.units = units
        self.T = T
        self.activation = tf.keras.activations.tanh
        self.W_width=self.add_weight(name='attention_weight_width', shape=(self.T,1), initializer='random_normal', trainable=True)
        self.b_width=self.add_weight(name='attention_bias_width', shape=(self.units,1), initializer='zeros', trainable=True)
        self.W_temporal=self.add_weight(name='attention_weight_temporal', shape=(self.units,1), initializer='random_normal', trainable=True)
        self.b_temporal=self.add_weight(name='attention_bias_temporal', shape=(self.T,1), initializer='zeros', trainable=True)

    def call(self,inputs):
        
        inputs = tf.stack(inputs, axis=1) #[bs,T,d]

        # width attention mechanism
        inputs = tf.transpose(inputs, [0,2,1]) #[bs,d,T]
        # Alignment scores. Pass them through tanh function
        h = K.dot(inputs, self.W_width) + self.b_width #[bs,d,1]
        h = self.activation(h) # [bs,d,1]
        alpha = K.softmax(K.squeeze(h,axis=-1))
        alpha = K.expand_dims(alpha, axis=-1) # [bs,d,1]
        context = inputs * alpha # [bs ,d, T]
        filtered_inputs = tf.transpose(context, [0,2,1]) #[bs,T,d]
        
        # temporal attention mechanism
        # Alignment scores. Pass them through tanh function
        h = K.dot(filtered_inputs, self.W_temporal) + self.b_temporal #[bs,T,1]
        h = self.activation(h) # [bs,T,1]
        alpha = K.softmax(K.squeeze(h,axis=-1))
        alpha = K.expand_dims(alpha, axis=-1) # [bs,T,1]
        context = filtered_inputs * alpha # [bs ,100, 4]
        context = K.sum(context, axis=1) #[bs, 4]
      
        return context

"""
(temporal + width) attention layer class (simple attention mechanism)
"""
class TemporalWidthAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,units, T):
        super(TemporalWidthAttentionLayer,self).__init__()
        self.units = units
        self.T = T
        self.activation = tf.keras.activations.tanh
        self.W_temporal=self.add_weight(name='attention_weight_temporal', shape=(self.units,1), initializer='random_normal', trainable=True)
        self.b_temporal=self.add_weight(name='attention_bias_temporal', shape=(self.T,1), initializer='zeros', trainable=True)
        self.W_width=self.add_weight(name='attention_weight_width', shape=(self.T,1), initializer='random_normal', trainable=True)
        self.b_width=self.add_weight(name='attention_bias_width', shape=(self.units,1), initializer='zeros', trainable=True)

    def call(self,inputs):
        
        inputs = tf.stack(inputs, axis=1) #[bs,T,d]

        # temporal attention mechanism
        # Alignment scores. Pass them through tanh function
        h = K.dot(inputs, self.W_temporal) + self.b_temporal #[bs,T,1]
        h = self.activation(h) # [bs,T,1]
        alpha = K.softmax(K.squeeze(h,axis=-1))
        alpha = K.expand_dims(alpha, axis=-1) # [bs,T,1]
        filtered_inputs = inputs * alpha # [bs ,100, 4]

        # width attention mechanism
        filtered_inputs = tf.transpose(filtered_inputs, [0,2,1]) #[bs,d,T]
        # Alignment scores. Pass them through tanh function
        h = K.dot(filtered_inputs, self.W_width) + self.b_width #[bs,d,1]
        h = self.activation(h) # [bs,d,1]
        alpha = K.softmax(K.squeeze(h,axis=-1))
        alpha = K.expand_dims(alpha, axis=-1) # [bs,d,1]
        context = filtered_inputs * alpha # [bs ,d, T]
        context = tf.transpose(context, [0,2,1]) #[bs,T,d]
        context = K.sum(context, axis=1) #[bs, 4]
      
        return context


@tf.function
def recurrent_penalty(A, B, weight=1.0):
    d = tf.shape(A)[0]
    penalty = tf.linalg.norm(((A @ B) - tf.linalg.eye(d)) + ((B @ A) - tf.linalg.eye(d)))
    return penalty * weight

class TiltedCrossentropyLoss(tf.keras.losses.Loss):
    
    def __init__(self, beta=0.05):
        super(TiltedCrossentropyLoss, self).__init__()
        self.beta = beta
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(reduction='none')
        self.name = 'tilted_crossentropy_loss'
        
    def call(self, y_true, y_pred):
        scaled_cross_entropy_loss = self.beta * self.bce_loss(y_true, y_pred)
        tilted_loss = tf.math.log(tf.reduce_mean(tf.math.exp(scaled_cross_entropy_loss)))
    
        return (1/self.beta) * tilted_loss
    

"""
Coadjoint Class Implementation
"""
class CoadjointModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights, recurrent_penalty, rec_penalty_weight):
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.recurrent_penalty = recurrent_penalty
        self.rec_penalty_weight = rec_penalty_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.rec_penalty_tracker = tf.keras.metrics.Mean(name='rec-penalty')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')

        self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error', 'tilted_crossentropy_loss']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        super(CoadjointModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        x, y = data
   
        if self.lc_weights[-1] != 0.0:
            with tf.GradientTape() as t2:
                with tf.GradientTape(persistent=True) as t1:
                    outputs = self(x, training=True)
                    L = self.loss_fn(y, outputs[0])
                    #L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
                    #L_total = L + L_rec
                    
                dL_dW = t1.gradient(L, self.trainable_variables)
                dL_dX = t1.gradient(L, outputs[1])
                G = self.adj_penalty(outputs[1], dL_dX) if self.adj_penalty == softmax_penalty_function else self.adj_penalty( dL_dX ) 

            dG_dW = t2.gradient(G, self.trainable_variables)
            #dG_dW = [tf.zeros(dL_dW[i].shape) if dG_dW[i] == None else dG_dW[i] for i in range(0,len(dG_dW))]
        
            del t1

            dL_plus_G_dW = [
                tf.add( x[0] * self.lc_weights[0], x[1] * self.lc_weights[1] )
                for x in zip(dL_dW, dG_dW)
                ]
            
            # update parameters
            self.optimizer.apply_gradients(zip(dL_plus_G_dW, self.trainable_variables))

        else:
            
            with tf.GradientTape(persistent=True) as t1:
                outputs = self(x, training=True)
                L = self.loss_fn(y, outputs[0])
                #L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
                #L_total = L + L_rec
                
            dL_dW = t1.gradient(L, self.trainable_variables)
            dL_dX = t1.gradient(L, outputs[1])
            G = self.adj_penalty(outputs[1], dL_dX) if self.adj_penalty == softmax_penalty_function else self.adj_penalty( dL_dX ) 

            del t1

            # update parameters
            self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        λ1 = tf.reduce_mean(tf.norm(dL_dX[0]))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1]))

        #Update Metrics
        self.loss_tracker.update_state(L)
        #self.rec_penalty_tracker.update_state(L_rec)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                #"recurrent-penalty": self.rec_penalty_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.rec_penalty_tracker, self.penalty_tracker, self.accuracy_tracker,
                    self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        with tf.GradientTape() as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs[0])
            #L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
            #L_total = L + L_rec

        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs[1])
        G = self.adj_penalty( dL_dX ) 
        #Compute Adjoint Penalty
        #if self.adj_penalty == softmax_penalty_function:
        #    G = self.adj_penalty(outputs[1], dL_dX)
        #else:
        #    G = self.adj_penalty( dL_dX ) 

        #Update Metrics
        λ1 = tf.reduce_mean(norm(dL_dX[0]))
        λN = tf.reduce_mean(norm(dL_dX[-1]))
        self.loss_tracker.update_state(L)
        #self.rec_penalty_tracker.update_state(L_rec)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                #"recurrent-penalty": self.rec_penalty_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

def input_contribution_penalty(W, R, V, time_horizon):

    out = []
    
    for i in range(0, time_horizon):
        W_power = tf.eye(W.shape[0])
        for j in range(0, time_horizon - i):
            W_power = W @ W_power  
        
        out.append(tf.squeeze(V @ W_power @ R, axis=-1))
    out = tf.expand_dims(tf.concat(out, axis=0),axis=-1)
    out = out @ tf.transpose(out)
    
    #out_sm = K.softmax(out)
    G = tf.math.reduce_variance(out)
    return G

def softmax_penalty_function(states, adjoints):

    N = len(adjoints)

    G0 = (norm(adjoints[-1]) - norm(adjoints[0])) ** 2

    mu_states = tf.squeeze(tf.stack([norm(i) for i in states], axis=1),axis=-1) # 32, 784
    mu_adjoints = tf.squeeze(tf.stack([norm(i) for i in adjoints], axis=1),axis=-1)

    sm_states = K.softmax(mu_states, axis=-1)
    sm_adjoints = K.softmax(mu_adjoints, axis=-1) # (B,T)
    
    G = tf.zeros(sm_states.shape[0]) #32

    for i in range(0,N):
        G += 1e2 * (sm_states[:,i] - sm_adjoints[:,i]) ** 2
        
   
    return tf.reduce_mean(G) + tf.reduce_mean(G0) 

def constrained_states_loss(states):
    T = tf.cast(len(states),dtype=tf.dtypes.float32)
    states = tf.stack(states,axis=1)
    obs_sum = tf.reduce_sum(states,axis=-1) # (BS, T)
    obs_sum = tf.linalg.norm(obs_sum, axis=-1)
    constraint = (obs_sum - tf.math.sqrt(T)) ** 2
    return tf.reduce_mean(constraint)

@tf.function
def adjoint_attention_penalty(adjoints, alpha):
    N = len(adjoints)
    alpha_adjoints = tf.unstack(tf.stack(adjoints, axis=1) * alpha, axis=1)

    nrms = [norm(i) for i in adjoints]
    alpha_nrms = [norm(i) for i in alpha_adjoints]

    G = tf.zeros(nrms[0].shape)

    for i in range(0,N):
        G += (nrms[i] - alpha_nrms[i]) ** 2

    return tf.reduce_mean(G)

@tf.function
def recurrent_state_constraint(states):
    sum_states = [tf.reduce_sum(i, axis=-1) for i in states]
    constraint = tf.zeros(shape=sum_states[0].shape[0])
    penalty = [(i-constraint)**2 for i in sum_states]
    return tf.reduce_mean(penalty)

@tf.function
def recurrent_field_constraint(states):
    states = tf.concat(states, axis=1)
    sum_states = tf.reduce_sum(states, axis=-1)
    constraint = -tf.ones(shape=sum_states.shape[0])
    penalty = (sum_states - constraint)**2
    return tf.reduce_mean(penalty)

@tf.function()
def recurrent_monic_penalty(weight_sequence):
    monic_sum = 0.0
    for i in weight_sequence:
        monic_sum += i
    penalty = (monic_sum + 1)**2

    return penalty

@tf.function()
def test_penalty(adjoints):
    N = int(len(adjoints) / 2)
    nrms = [norm(i) for i in adjoints]
    t1 = tf.zeros(nrms[0].shape)
    for i in range(0,N):
        t1 += nrms[i]

    t1 = t1 / N 

    G = tf.zeros(nrms[0].shape)
    for i in range(0,N):
        G += nrms[i] ** 2

    return tf.reduce_mean(G) #- tf.reduce_mean(G_adj)

"""
Jacobian Class Implementation
"""
class JacobianPenaltyModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights, recurrent_penalty, rec_penalty_weight, full_gradients=None):
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.recurrent_penalty = recurrent_penalty
        self.full_gradients = full_gradients
        self.rec_penalty_weight = rec_penalty_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.rec_penalty_tracker = tf.keras.metrics.Mean(name='recurrent-penalty')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')
        #self.mse_loss = tf.keras.losses.MeanSquaredError()
        #self.bce_loss = tf.keras.losses.BinaryCrossentropy(reduction='none')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error', 'tilted_crossentropy_loss']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        
        super(JacobianPenaltyModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        x, y = data
   
        if self.lc_weights[-1] != 0.0:
        
            with tf.GradientTape() as t2:
                with tf.GradientTape(persistent=True) as t1:
                    outputs = self(x, training=True) # (predictions, states, attention_state)
                    L = self.loss_fn(y, outputs[0])
                    L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
                    L_total = L + L_rec
            
                dL_dW = t1.gradient(L_total, self.trainable_variables)
                dL_dX = t1.gradient(outputs[0], outputs[1])
                G = self.adj_penalty(dL_dX)
                #G = softmax_penalty_function(outputs[1], dL_dX)
             
            dG_dW = t2.gradient(G, self.trainable_variables)
            #dL_dW = [tf.zeros(dL_dW[i].shape) if i >= 4 else dL_dW[i] for i in range(0,len(dL_dW))]
            dG_dW = [tf.zeros(dG_dW[i].shape) if dG_dW[i] == None else dG_dW[i] for i in range(0,len(dL_dW))] 
            
            del t1
           
            dL_plus_G_dW = [
                tf.add( x[0] * self.lc_weights[0], x[1] * self.lc_weights[1] )
                for x in zip(dL_dW, dG_dW)
                ]
            
            # update parameters
            self.optimizer.apply_gradients(zip(dL_plus_G_dW, self.trainable_variables))

        else:
            with tf.GradientTape(persistent=True) as t1:
                outputs = self(x, training=True) # (predictions, states, attention_state)
                L = self.loss_fn(y, outputs[0])
                L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
                L_total = L + L_rec

            dL_dW = t1.gradient(L_total, self.trainable_variables)
            dL_dX = t1.gradient(outputs[0], outputs[1])
            G = self.adj_penalty(dL_dX)

            del t1
            
            # update parameters
            if self.full_gradients != None:
                dL_dW = [dL_dW[i] - self.full_gradients[i] for i in range(0,len(dL_dW))]

            self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        λ1 = tf.reduce_mean(tf.norm(dL_dX[0]))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1]))

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.rec_penalty_tracker.update_state(L_rec)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "recurrent-penalty": self.rec_penalty_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.rec_penalty_tracker, self.penalty_tracker, self.accuracy_tracker,
                    self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        
        with tf.GradientTape(persistent=True) as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs[0])
            L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
            
        #Compute Adjoints and Adjoint Penalty
        dL_dX = t1.gradient(outputs[0], outputs[1])
        G = self.adj_penalty(dL_dX)
        #G = softmax_penalty_function(outputs[1], dL_dX)
     
        del t1
        #Update Metrics
        λ1 = tf.reduce_mean(norm(dL_dX[0]))
        λN = tf.reduce_mean(norm(dL_dX[-1]))
        self.loss_tracker.update_state(L)
        self.rec_penalty_tracker.update_state(L_rec)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "recurrent-penalty": self.rec_penalty_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

"""
Jacobian Class Implementation
"""
class InitializationModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights, recurrent_penalty, rec_penalty_weight, full_gradients=None):
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.recurrent_penalty = recurrent_penalty
        self.full_gradients = full_gradients
        self.rec_penalty_weight = rec_penalty_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.rec_penalty_tracker = tf.keras.metrics.Mean(name='recurrent-penalty')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(reduction='none')
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        
        super(InitializationModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        x, y = data
   
        if self.lc_weights[-1] > 0.0:
        
            with tf.GradientTape() as t2:
                with tf.GradientTape(persistent=True) as t1:
                    outputs = self(x, training=True) # (predictions, states, attention_state)
                    L = self.loss_fn(y, outputs[0])
                    L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
                    L_total = L + L_rec
            
                dL_dW = t1.gradient(L_total, self.trainable_variables)
                dL_dX = t1.gradient(outputs[0], outputs[1])
                G = self.adj_penalty(dL_dX)
                #G = softmax_penalty_function(outputs[1], dL_dX)
             
            dG_dW = t2.gradient(G, self.trainable_variables)
            dG_dW = [tf.zeros(dG_dW[i].shape) if dG_dW[i] == None else dG_dW[i] for i in range(0,len(dL_dW))] 
            
            del t1
           
            dL_plus_G_dW = [
                tf.add( x[0] * self.lc_weights[0], x[1] * self.lc_weights[1] )
                for x in zip(dL_dW, dG_dW)
                ]
            
            # update parameters
            self.optimizer.apply_gradients(zip(dL_plus_G_dW, self.trainable_variables))

        else:
            with tf.GradientTape() as t2:
                with tf.GradientTape(persistent=True) as t1:
                    outputs = self(x, training=True) # (predictions, states, attention_state)
                    L = self.loss_fn(y, outputs[0])
                    L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
                    L_total = L + L_rec

                #dL_dW = t1.gradient(L_total, self.trainable_variables)
                dL_dX = t1.gradient(L, outputs[1])
                G = self.adj_penalty(dL_dX)

                del t1
            dG_dW = t2.gradient(G, self.trainable_variables)
                
            self.optimizer.apply_gradients(zip(dG_dW, self.trainable_variables))

        λ1 = tf.reduce_mean(tf.norm(dL_dX[0]))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1]))

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.rec_penalty_tracker.update_state(L_rec)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "recurrent-penalty": self.rec_penalty_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.rec_penalty_tracker, self.penalty_tracker, self.accuracy_tracker,
                    self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        
        with tf.GradientTape(persistent=True) as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs[0])
            L_rec = recurrent_penalty(self.layers[1].W1, self.layers[1].W2, weight=self.rec_penalty_weight) if self.recurrent_penalty else 0.0
            
        #Compute Adjoints and Adjoint Penalty
        dL_dX = t1.gradient(outputs[0], outputs[1])
        G = self.adj_penalty(dL_dX)
        #G = softmax_penalty_function(outputs[1], dL_dX)
     
        del t1
        #Update Metrics
        λ1 = tf.reduce_mean(norm(dL_dX[0]))
        λN = tf.reduce_mean(norm(dL_dX[-1]))
        self.loss_tracker.update_state(L)
        self.rec_penalty_tracker.update_state(L_rec)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "recurrent-penalty": self.rec_penalty_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

"""
Reservoir Class Implementation
"""
class ReservoirPenaltyModel(tf.keras.Model):

    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights, recurrent_penalty, rec_penalty_weight, full_gradients=None):
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.recurrent_penalty = recurrent_penalty
        self.full_gradients = full_gradients
        self.rec_penalty_weight = rec_penalty_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.rec_penalty_tracker = tf.keras.metrics.Mean(name='recurrent-penalty')
        
        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        

        super(ReservoirPenaltyModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):

        x, y = data

        with tf.GradientTape(persistent=True) as t1:
            outputs = self(x, training=True)
            L = self.loss_fn(y, outputs[0])
        

        dL_dW = t1.gradient(L, self.trainable_variables)        
    
        del t1

        # update parameters
        if self.full_gradients != None:
            dL_dW = [dL_dW[i] - self.full_gradients[i] for i in range(0,len(dL_dW))]

        self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.accuracy_tracker.update_state(y,outputs[0])
       

        return {"loss": self.loss_tracker.result(),
                "accuracy": self.accuracy_tracker.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        with tf.GradientTape() as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs[0])

        #Update Metrics
        self.loss_tracker.update_state(L)
 
        self.accuracy_tracker.update_state(y,outputs[0])

        return {"loss": self.loss_tracker.result(),
                "accuracy": self.accuracy_tracker.result()
                }

"""
Creates coadjoint model from provided arguments
"""
def make_model(name, T, ft_dim, hid_dim, out_dim, penalty_weight, learning_rate, batch_size=32, penalty_type='coadjoint', attention=False, invert_index=None, rec_penalty_weight=1.0, beta=0.0):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    recurrent_penalty = True if name == 'inversepenaltyrnn' else False
    output_activation = 'softmax' if out_dim > 1 else 'sigmoid'
    loss = tf.keras.losses.SparseCategoricalCrossentropy() if out_dim > 1 else tf.keras.losses.BinaryCrossentropy()
    #if beta == 0:
    #    loss = tf.keras.losses.SparseCategoricalCrossentropy() if out_dim > 1 else tf.keras.losses.BinaryCrossentropy()
    #else:
    #    loss = TiltedCrossentropyLoss(beta=beta)

    layers = {'rnn': RNNRecurrentLayer(hid_dim, ft_dim, T),
              'componentrnn': ComponentRecurrentLayer(hid_dim, ft_dim, T),
              'testrnn': TestRecurrentLayer(hid_dim, ft_dim, T),
              'tiltedrnn': TiltedRecurrentLayer(hid_dim, ft_dim, T, beta),
              'linearrnn': LinearRecurrentLayer(hid_dim, ft_dim, T),
              'monicrnn': MonicRNNRecurrentLayer(hid_dim, ft_dim, T),
              'normrnn': NormRNNRecurrentLayer(hid_dim, ft_dim, T),
              'newrnn': NewRNNRecurrentLayer(hid_dim, ft_dim, T),
              'filteringrnn': FilteringRNNRecurrentLayer(hid_dim, ft_dim, T),
              'inputattentionrnn': InputAttentionRNNRecurrentLayer(hid_dim, ft_dim, T),
              'threadrnn': ThreadLayer(hid_dim, ft_dim, T),
              'reservoir': ReservoirLayer(hid_dim, ft_dim, T),
              'transportrnn': TransportRNNRecurrentLayer(hid_dim, ft_dim, T),
              'dynamicalrnn': DynamicalRNNRecurrentLayer(hid_dim, ft_dim, T),
              'fusernn': FuseRNNRecurrentLayer(hid_dim, ft_dim, T),
              'shieldedrnn': ShieldedRNNRecurrentLayer(hid_dim, ft_dim, T),
              'doublernn': DoubleRNNRecurrentLayer(hid_dim, ft_dim, T),
              'linearstatesrnn': LinearStatesRNNRecurrentLayer(hid_dim, ft_dim, T),
              'constrainedrnn': ConstrainedRNNRecurrentLayer(hid_dim, ft_dim, T),
              'compressionrnn': CompressionRNNRecurrentLayer(hid_dim, ft_dim, T),
              'lurnn': LURNNRecurrentLayer(hid_dim, ft_dim, T, invert_index),
              'fixedinputrnn': FixedInputRNNRecurrentLayer(hid_dim, ft_dim, T),
              'fixedrecurrentrnn': FixedRecurrentRNNRecurrentLayer(hid_dim, ft_dim, T),
              'inversepenaltyrnn': InversePenaltyRNNRecurrentLayer(hid_dim, ft_dim, T, invert_index),
              'exactinversernn': ExactInverseRNNRecurrentLayer(hid_dim, ft_dim, T, invert_index),
              'covariancernn': CovarianceRNNRecurrentLayer(hid_dim, ft_dim, T),
              'transposernn': TransposeRNNRecurrentLayer(hid_dim, ft_dim, T),
              'lstm': LSTMLayer(hid_dim),
              'gru': GRULayer(hid_dim),
              'antisymmetric': antiRNNLayer(hid_dim, ft_dim, epsilon=0.01, gamma=0.01, sigma=0.01)
             }
    
    attention_layer_dict = {'temporal': TemporalAttentionLayer(hid_dim, T), 
                       'width': WidthAttentionLayer(hid_dim, T),
                       'temporalwidth': TemporalWidthAttentionLayer(hid_dim, T),
                       'widthtemporal': WidthTemporalAttentionLayer(hid_dim, T)
                       }
    
    inputs = tf.keras.Input(shape=(T,ft_dim), batch_size=batch_size, name='input-layer')

    attention_layer = attention_layer_dict[attention] if attention != False else None
    rec_layer = layers[name]   
    dense_layer = tf.keras.layers.Dense(out_dim, activation=output_activation, name='output-layer', trainable=True)
    
    if attention != False:
        states = rec_layer(inputs)
        (attention_state, anti_attention_state) = attention_layer(states)
        outputs = dense_layer(attention_state)
        outputs2 = dense_layer(anti_attention_state)
    elif name in ['linearstatesrnn']:
        (states, linear_states) = rec_layer(inputs)
        outputs = dense_layer(states[-1])
    elif name == 'compressionrnn':
        (states,compression_states) = rec_layer(inputs)
        outputs = dense_layer(compression_states[-1])
    elif name == 'secondorderrnn':
        rec_layer_2 = RNNRecurrentLayer(hid_dim, hid_dim, T)
        states = rec_layer(inputs)
        state_diffs = tf.stack([states[i+1] - states[i] for i in range(0,len(states)-1)],axis=1)
        states_2 = rec_layer_2(state_diffs)
        outputs = dense_layer(tf.concat((states[-1],states_2[-1]),axis=-1))
    elif name == 'threadrnn':
        rnn_rec_layer = RNNRecurrentLayer(hid_dim, ft_dim, T)
        thread_states = rec_layer(inputs)
        states = rnn_rec_layer(inputs)
        merge_states = tf.concat((states[-1], thread_states), axis=-1)
        outputs = dense_layer(merge_states)
    elif name == 'inputattentionrnn':
        attn_layer = TemporalAttentionLayer(hid_dim, T)
        (states, original_states, alpha) = rec_layer(inputs)
        attn_states = attn_layer(states)
        outputs = dense_layer(attn_states)
        #outputs = dense_layer(states[-1])
    elif name == 'reservoirattentionrnn':
        states = rec_layer(inputs)
        outputs = dense_layer(states)
    elif name == 'newrnn':
        (states, pos_states, neg_states) = rec_layer(inputs)
        outputs = dense_layer(states)
    else:
        states = rec_layer(inputs)
        outputs = dense_layer(states[-1])

    if penalty_type == 'coadjoint':
        penalty = scaled_variance_adjoint_penalty_L2
        if name == 'compressionrnn':
            model = CoadjointModel(inputs=inputs, outputs=[outputs, states, compression_states])
        elif name == 'inputattentionrnn':
            model = CoadjointModel(inputs=inputs, outputs=[outputs, states, original_states, alpha])
        else:
            model = CoadjointModel(inputs=inputs, outputs=[outputs, states, attention_state]) if attention != False else CoadjointModel(inputs=inputs, outputs=[outputs, states])
    elif penalty_type == 'jacobian':
        penalty = input_output_jacobian_norm_variance_penalty_L2
        #penalty = adjoint_attention_penalty
        if name == 'linearstatesrnn':
            model = JacobianPenaltyModel(inputs=inputs, outputs=[outputs, states, linear_states])
        elif name == 'newrnn':
            model = JacobianPenaltyModel(inputs=inputs, outputs=[outputs, states, pos_states, neg_states])
        else:
            model = JacobianPenaltyModel(inputs=inputs, outputs=[outputs, states, attention_state, outputs2]) if attention != False else JacobianPenaltyModel(inputs=inputs, outputs=[outputs, states])
    elif penalty_type == 'reservoir':
        penalty = input_output_jacobian_norm_variance_penalty_L2
        model = ReservoirPenaltyModel(inputs=inputs, outputs=[outputs, states])
    elif penalty_type == 'initialization':
        penalty = scaled_variance_adjoint_penalty_L2
        model = InitializationModel(inputs=inputs, outputs=[outputs, states])
    
    model.compile(optimizer=optimizer,
            loss_fn=loss,
            lc_weights= [penalty_weight[0], penalty_weight[1]],
            adj_penalty=penalty,
            model_name=name,
            recurrent_penalty=recurrent_penalty,
            rec_penalty_weight=rec_penalty_weight
            )

        
    return model