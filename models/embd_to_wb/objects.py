import numpy as np
from tensorflow import keras
import tensorflow as tf

from typing import Union, List


class EncoderLayer(keras.layers.Layer):
    """
    encodes vector into sequence
    """
    def __init__(self, alphabet_size: int, max_word_length: int = 16, rnn_units: int = 128, **kwargs):
        """
        :param alphabet_size: letters in target alphabet
        :param max_word_length: maximal length of generated words
        :param rnn_units: hidden units in rnn_units network
        :param kwargs: kwargs
        """
        super().__init__(**kwargs)
        self.repeat = keras.layers.RepeatVector(max_word_length)
        self.rnn = keras.layers.SimpleRNN(units=rnn_units, return_sequences=True)
        self.dense = keras.layers.Dense(alphabet_size, activation='softmax')

    def call(self, inputs, *args, **kwargs):
        x = self.repeat(inputs)
        x = self.rnn(x)
        return self.dense(x)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "alphabet_size": self.dense.units,
            "max_word_length": self.repeat.n,
            "rnn_units": self.rnn.units
        })
        return config


class EncoderLayerSyl(keras.layers.Layer):
    """
    generalisation of EncoderLayer
    encodes input vector into sequence of probabilities, generates output sequence in blocks
    """
    def __init__(self, alphabet_size: int, syllables_len: int = 2, syllables: int = 8, rnn_units: int = 128, **kwargs):
        """
        :param alphabet_size: size of target alphabet
        :param syllable_len: length of single generated block (syllable) of sequence
        :param syllables: number of blocks (syllables) to generate (note that if syllables_len=1,
         layer acts like normal EncoderLayer)
        :param rnn_units: hidden units of rnn sublayer
        :param kwargs: kwargs (idk what else can i say)
        """
        super(EncoderLayerSyl, self).__init__(**kwargs)
        self.alphabet_size = alphabet_size
        self.syllables_len = syllables_len
        self.syllables = syllables
        self.rnn_units = rnn_units
        self.repeat = keras.layers.RepeatVector(syllables)
        self.rnn = keras.layers.SimpleRNN(rnn_units, return_sequences=True)
        self.dense = keras.layers.Dense(syllables_len * alphabet_size)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs, *args, **kwargs):
        x = self.repeat(inputs)
        x = self.rnn(x)
        x = self.dense(x)
        x = tf.reshape(x, [-1, self.syllables_len * self.syllables, self.alphabet_size])
        return self.softmax(x)

    def get_config(self):
        config = super(EncoderLayerSyl, self).get_config()
        config.update({
            "syllables": self.syllables,
            "syllables_len": self.syllables_len,
            "rnn_units": self.rnn_units,
            "alphabet_size": self.alphabet_size
        })
        return config


class FiltrationLayer(keras.layers.Layer):
    """
    clears sequence leaving only top k probabilities for each letter
    this is supposed to reduce probability of encoder to encoder input vector by simply slicing it into sequence
    """

    def __init__(self, probabilities_to_keep: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.k = probabilities_to_keep

    def call(self, inputs, *args, **kwargs):
        """pivot = tf.reduce_min(tf.math.top_k(inputs, self.k + 1).values, axis=-1)
        mask = tf.repeat(tf.expand_dims(pivot, -1), inputs.shape[-1], axis=-1) < inputs
        return self.softmax(inputs * tf.cast(mask, tf.float32))"""
        pivot = tf.reduce_min(tf.math.top_k(inputs, self.k).values, axis=-1)
        mask = tf.repeat(tf.expand_dims(pivot, -1), inputs.shape[-1], axis=-1) > inputs
        return (inputs - tf.stop_gradient(inputs) * tf.cast(mask, tf.float32)) + (1 - tf.stop_gradient(inputs)) * (1 - tf.cast(mask, tf.float32))

    def get_config(self):
        config = super(FiltrationLayer, self).get_config()
        config.update({
            "probabilities_to_keep": self.k
        })
        return config


class DecoderLayer(keras.layers.Layer):
    def __init__(self, vector_dim: int, ds_mean: np.ndarray, ds_variance: np.ndarray, rnn_units: int = 128, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.rnn = keras.layers.SimpleRNN(units=rnn_units)
        self.dense = keras.layers.Dense(units=vector_dim)
        self.normalization = keras.layers.Normalization(mean=ds_mean, variance=ds_variance)

    def call(self, inputs, *args, **kwargs):
        x = self.rnn(inputs)
        x = self.dense(x)
        return self.normalization(x)

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            "vector_dim": self.dense.units,
            "ds_mean": self.normalization.mean.numpy(),
            "ds_variance": self.normalization.variance.numpy(),
            "rnn_units": self.rnn.units
        })
        return config


class DecoderLayerSyl(keras.layers.Layer):
    """
    generalisation of DecoderLayer
    decodes vector from discrete sequence, analyzes in blocks
    """
    def __init__(self,
                 vector_dim: int,
                 ds_mean: np.ndarray,
                 ds_variance: np.ndarray,
                 rnn_units: int = 128,
                 syllables_len: int = 2,
                 **kwargs):
        """
        :param vector_dim: dimension of target vector space
        :param ds_mean: vector space mean
        :param ds_variance: vector space variance
        :param rnn_units: rnn sublayer hidden units
        :param syllable_len: length of syllable
        :param kwargs: kwargs
        """
        super(DecoderLayerSyl, self).__init__(**kwargs)
        self.vector_dim = vector_dim
        self.ds_mean = ds_mean
        self.ds_variance = ds_variance
        self.rnn_units = rnn_units
        self.syllable_len = syllables_len
        self.rnn = keras.layers.SimpleRNN(self.rnn_units)
        self.dense = keras.layers.Dense(self.vector_dim)
        self.normalization = keras.layers.Normalization(mean=self.ds_mean, variance=self.ds_variance)

    def call(self, inputs, *args, **kwargs):
        x = tf.reshape(inputs, [-1, inputs.shape[-2] // self.syllable_len, self.syllable_len * inputs.shape[-1]])
        x = self.rnn(x)
        x = self.dense(x)
        return self.normalization(x)

    def get_config(self):
        config = super(DecoderLayerSyl, self).get_config()
        config.update({
            "vector_dim": self.vector_dim,
            "ds_mean": self.ds_mean,
            "ds_variance": self.ds_variance,
            "rnn_units": self.rnn_units,
            "syllables_len": self.syllable_len
        })
        return config


class ConsistencyLoss(keras.losses.Loss):
    """
    penalizes to small gain in EOW token probability over sequence length
    """
    def __init__(self, end_of_word_ind: int, gain_per_step: float = 0.05,  **kwargs):
        """
        :param end_of_word_ind: index of end-of-word feature in letter probability vector
        :param gain_per_step: minimal not penalized gain
        :param kwargs: kwargs
        """
        self.EOW_ind = end_of_word_ind
        self.gain_per_step = gain_per_step
        super(ConsistencyLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.relu(
            y_pred[..., :-1, self.EOW_ind] - y_pred[..., 1:, self.EOW_ind] + self.gain_per_step), axis=-1)

    def get_config(self):
        config = super(ConsistencyLoss, self).get_config()
        config.update({
            "end_of_word_ind": self.EOW_ind,
            "gain_per_step": self.gain_per_step
        })
        return config


class LettersDistributionLoss(keras.losses.Loss):
    """penalizes deviation from given probabilities of probabilities P(c(i+1)|c(i))
     where c(i) is i-th character in word"""

    def __init__(self, probabilities: Union[np.ndarray, tf.Tensor], **kwargs):
        """
        :param probabilities: matrix of probabilities where probabilities[i][j] is probability that n+1-th letter will
        be indexed with j given that n-th letter is indexed with i
        :param kwargs: kwargs
        """
        super(LettersDistributionLoss, self).__init__(**kwargs)
        self.p = probabilities if isinstance(probabilities, tf.Tensor) else tf.convert_to_tensor(probabilities)
        self.p = tf.cast(self.p, tf.float32)

    def call(self, y_true, y_pred):
        # todo: temp solution, assumes that EOW_token is always at the end of char vector
        return tf.reduce_mean(keras.losses.categorical_crossentropy(
            (1 - tf.cast(y_pred[..., :-1, :-1] < tf.repeat(tf.expand_dims(tf.reduce_max(y_pred[..., :-1, :-1], axis=-1), axis=-1), y_pred.shape[-1] - 1, axis=-1), dtype=tf.float32)) @ self.p,
            y_pred[..., 1:, :-1]
        ))

    def get_config(self):
        config = super(LettersDistributionLoss, self).get_config()
        config.update({
            "probabilities": self.p.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        p = config.pop("probabilities")
        p = tf.convert_to_tensor(p)
        return cls(probabilities=p, **config)


class SequenceLoss(keras.losses.Loss):
    """combines all losses for output of encoder"""

    def __init__(self, consistency_loss_kwargs: dict, letters_distribution_loss_kwargs: dict, weights: dict, **kwargs):
        """
        :param consistency_loss_kwargs: args for ConsistencyLoss instance
        :param letters_distribution_loss_kwargs: args for LettersDistributionLoss instance
        :param weights: weights for losses (should contain "consistency_loss" and "letters_distribution_loss" keys
        :param kwargs: kwargs
        """
        super(SequenceLoss, self).__init__(**kwargs)
        self.ldl = LettersDistributionLoss(**letters_distribution_loss_kwargs)
        self.cl = ConsistencyLoss(**consistency_loss_kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        return self.ldl(y_true, y_pred) * self.weights["letters_distribution_loss"] +\
               self.cl(y_true, y_pred) * self.weights["consistency_loss"]

    def get_config(self):
        config = super(SequenceLoss, self).get_config()
        config.update({
            "consistency_loss_kwargs": self.cl.get_config(),
            "letters_distribution_loss_kwargs": self.ldl.get_config(),
            "weights": self.weights
        })
        return config
