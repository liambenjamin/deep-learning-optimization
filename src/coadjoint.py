import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class CoadjointModel(tf.keras.Model):
    """Coadjoint Model Class

    Args:
        1. model_name (string):    Recurrent layer name (e.g., "rnn", "lstm", "gru")
        2. optimizer (tf.keras.optimizer):     Optimizer used during training
        3. loss_fn (tf.keras.losses object):   Loss function
        4. adj_penalty:            adjoint regularizer function (import "adjoint_regularizers.py")
        5. lc_weights (list of 2): [weight applied to objective F, weight applied to objective G]

    Returns: Coadjoint model
       
    
    """
    def compile(self, model_name, optimizer, loss_fn, adj_penalty, lc_weights):
        """compile coadjoint model"""
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.adj_penalty = adj_penalty
        self.lc_weights = lc_weights
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.penalty_tracker = tf.keras.metrics.Mean(name='penalty')
        self.first_adjoint = tf.keras.metrics.Mean(name='first-adjoint')
        self.last_adjoint = tf.keras.metrics.Mean(name='last-adjoint')

        if self.loss_fn.name in ['binary_crossentropy', 'mean_squared_error', 'tilted_crossentropy_loss']:
            self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        else:
            self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        super(CoadjointModel, self).compile(optimizer=optimizer)

    @tf.function
    def train_step(self, data):
        """Training step for coadjoint regularized model

        Args:
            data: training batch (x,y)

        Returns:
            training metrics recorded
                1. Loss (F)
                2. Accuracy
                3. Penalty (G)
                4. Initial and terminal adjoint lengths
        """
        x, y = data
   
        if self.lc_weights[-1] != 0.0: # perform coadjoint update
            with tf.GradientTape() as t2:
                with tf.GradientTape(persistent=True) as t1:
                    outputs = self(x, training=True)
                    L = self.loss_fn(y, outputs[0])
                    
                dL_dW = t1.gradient(L, self.trainable_variables)
                dL_dX = t1.gradient(L, outputs[1])
                G = self.adj_penalty(dL_dX)

            dG_dW = t2.gradient(G, self.trainable_variables)
            
            del t1

            dL_plus_G_dW = [
                tf.add( x[0] * self.lc_weights[0], x[1] * self.lc_weights[1] )
                for x in zip(dL_dW, dG_dW)
                ]
            
            self.optimizer.apply_gradients(zip(dL_plus_G_dW, self.trainable_variables))

        else: #perform adjoint update (backpropagation)
            
            with tf.GradientTape(persistent=True) as t1:
                outputs = self(x, training=True)
                L = self.loss_fn(y, outputs[0])
                
            dL_dW = t1.gradient(L, self.trainable_variables)
            dL_dX = t1.gradient(L, outputs[1])
            G = self.adj_penalty( dL_dX ) 

            del t1

            self.optimizer.apply_gradients(zip(dL_dW, self.trainable_variables))

        λ1 = tf.reduce_mean(tf.norm(dL_dX[0], axis=1))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1], axis=1))

        #Update Metrics
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }

    @property
    def metrics(self):
        return [self.loss_tracker, self.penalty_tracker, self.accuracy_tracker, 
                self.first_adjoint, self.last_adjoint]

    @tf.function
    def test_step(self, data):
        """Test/evaluation routine for coadjoint model"""

        x, y = data

        with tf.GradientTape() as t1:
            outputs = self(x)
            L = self.loss_fn(y, outputs[0])

        #Compute Adjoints
        dL_dX = t1.gradient(L, outputs[1])
        G = self.adj_penalty(dL_dX) 

        #Update Metrics
        λ1 = tf.reduce_mean(tf.norm(dL_dX[0], axis=1))
        λN = tf.reduce_mean(tf.norm(dL_dX[-1], axis=1))
        self.loss_tracker.update_state(L)
        self.penalty_tracker.update_state(G)
        self.accuracy_tracker.update_state(y,outputs[0])
        self.first_adjoint.update_state(λ1)
        self.last_adjoint.update_state(λN)

        return {"loss": self.loss_tracker.result(),
                "penalty": self.penalty_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "λ1": self.first_adjoint.result(),
                "λN": self.last_adjoint.result()
                }
