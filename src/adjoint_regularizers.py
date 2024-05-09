import numpy as np
import tensorflow as tf


@tf.custom_gradient
def norm(x): #x (bs, hid_dim)
    """Custom gradient for tf.norm (L2)

    Args: tensor

    Returns: L2 norm of tensor
    """
    ϵ = 1.0e-17
    nrm = tf.norm(x, axis=1, keepdims=True)

    def grad(dy):
        return dy * tf.math.divide(x,(nrm + ϵ))
    
    return nrm, grad


@tf.function
def scaled_variance_adjoint_penalty_L2(adjoints):
    """Adjoint Penalty: scaled variance adjoint penalty (L2 norm)
    
    Args: 
        adjoints: list of tensors with dimension [batch,d]

    Returns:
        G: scalar adjoint penalty (averaged over batch)
    """
    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]
    G = tf.zeros(nrms[0].shape)
    t1 = tf.zeros(nrms[0].shape)

    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N

    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)


@tf.function
def scaled_variance_adjoint_penalty_L1(adjoints):
    """Adjoint Penalty: scaled variance adjoint penalty (L1 norm)

    Args: 
        adjoints: list of tensors with dimension [batch,d]

    Returns:
        G: scalar adjoint penalty (averaged over batch)
    """
    N = len(adjoints)
    nrms = [tf.norm(adjoints[i], axis=1, ord=1) for i in range(0,N)]
    G = tf.zeros(nrms[0].shape)
    t1 = tf.zeros(nrms[0].shape)
    
    for i in range(0, N):
        t1 += nrms[i]

    t1 = t1 / N
    
    for i in range(0, N):
        G += (nrms[i] - t1) ** 2

    return tf.reduce_mean(G)


@tf.function
def pascanu_adjoint_penalty(adjoints):
    """Penalty from Pascanu et. 2013

    "On the difficulty of training recurrent neural networks"

    Args: 
        adjoints: list of tensors with dimension [batch,d]

    Returns:
        G: scalar adjoint penalty (averaged over batch)
    """
    N = len(adjoints)
    nrms = [norm(adjoints[i]) for i in range(0,N)]
    G = tf.zeros(nrms[0].shape)

    for i in range(0,N-1):
        G += (nrms[i] / nrms[i+1] - 1) ** 2

    return tf.reduce_mean(G)


@tf.function
def input_output_jacobian_norm_variance_penalty_L2(adjoints):
    """Adjoint Penalty: input-output jacobian variance adjoint penalty (L2 norm)
    
    G(l) = ( ||l_1|| - ||l_2|| )^2

    Args: 
        adjoints: list of tensors with dimension [batch,d]

    Returns:
        G: scalar adjoint penalty (averaged over batch)
    """
    G = (norm(adjoints[0]) - norm(adjoints[-1])) ** 2 
    
    return tf.reduce_mean(G)


