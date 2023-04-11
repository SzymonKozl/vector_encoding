from tensorflow import keras
from tensorflow import data as tf_data
import tensorflow as tf
import numpy as np

from scripts.loader import load as load_ds
from objects import EncoderLayerSyl, DecoderLayerSyl, ConsistencyLoss, FiltrationLayer, SequenceLoss


# data load & constants
vectors = load_ds()
DS_MEAN = np.mean(vectors, axis=0)
DS_VARIANCE = np.var(vectors, axis=0)
EMBEDDING_DIM = vectors.shape[1]
ALPHABET_SIZE = 27 # 26 lowercase english letters and <end_of_word>
EOW_INDEX = 26
MAX_WORD_LEN = 15
SYLLABLE_LEN = 3
vectors = tf.convert_to_tensor(vectors)
dataset = tf_data.Dataset.from_tensor_slices((vectors, vectors))

probabilities = np.load(r"C:\Users\szymon\Desktop\pycharm\babel\data\letter_probabilites.npy")


# training constants
BATCH_SIZE = 128
BATCHES_PER_EPOCH = 1024
EPOCHS = 150

# model hiperparameters
VALUES_TO_KEEP = 1
RNN_UNITS = 768


optimizer = keras.optimizers.experimental.Adafactor(beta_2_decay=-1.0)
encoder = EncoderLayerSyl(ALPHABET_SIZE, SYLLABLE_LEN, MAX_WORD_LEN // SYLLABLE_LEN, RNN_UNITS, name='encoder')
decoder = DecoderLayerSyl(EMBEDDING_DIM, DS_MEAN, DS_VARIANCE, RNN_UNITS, SYLLABLE_LEN, name='decoder')
filtration = FiltrationLayer(VALUES_TO_KEEP)
inp = keras.layers.Input((EMBEDDING_DIM,))
encoded = encoder(inp)
t = filtration(encoded)
y = decoder(t)

model = keras.models.Model(inputs=inp, outputs=(y, encoded))

"""
        'encoder': SequenceLoss(
            consistency_loss_kwargs={
                "end_of_word_ind": EOW_INDEX
            },
            letters_distribution_loss_kwargs={
                "probabilities": probabilities
            },
            weights={
                "letters_distribution_loss": 0.5,
                "consistency_loss": 0.5
            }
        )"""
model.compile(
    loss={
        'decoder': keras.losses.CosineSimilarity()
    },
    loss_weights={
        'decoder': 1.0
    },
    optimizer=optimizer
)

model.summary()
model.fit(dataset.batch(BATCH_SIZE).repeat(), epochs=EPOCHS, steps_per_epoch=BATCHES_PER_EPOCH)

model.save(r"C:\Users\szymon\Desktop\pycharm\babel\models\embd_to_wb\trained\m1_0syl", save_traces=False)
