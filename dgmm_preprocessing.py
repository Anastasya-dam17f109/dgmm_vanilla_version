import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from collections import defaultdict
import numpy as np
from ucimlrepo import fetch_ucirepo
import tensorflow as tf
import keras
from keras import layers
#import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


#num_classes = 10
#input_shape = (32, 32, 3)

valid_rate = 0.1
target_size = 32  # Resize the input images.
representation_dim = 512  # The dimensions of the features vector.
projection_units = 128  # The projection head of the representation learner.
num_clusters = 20  # Number of clusters.
k_neighbours = 5  # Number of neighbours to consider during cluster learning.
tune_encoder_during_clustering = False
uci_idx = [109]
train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

# fetch dataset
def download_from_uci(idx):
    uci_data = fetch_ucirepo(id=idx)
    X = uci_data.data.features
    y = uci_data.data.targets
    train_size = y.size - int(y.size*valid_rate)
    X = X.reindex(np.random.permutation(X.index))
    X =(X - X.min()) / (X.max() - X.min())
    X_train, X_test = X.iloc[0:train_size], X.iloc[train_size:]
    X_train.to_csv(train_data_file, index=False,  sep=';',)
    X_test.to_csv(test_data_file, index=False,  sep=';',)

def get_dataset_from_csv(csv_file_path,  shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path, field_delim=';',batch_size=32
        )

    return dataset

def preprocess_on_python():
    x_train = get_dataset_from_csv(train_data_file)
    x_test = get_dataset_from_csv(test_data_file)

download_from_uci(109)
#os.system('"C:\\Program Files\\R\\R-4.3.1\\bin\\Rscript.exe" C:\\Users\\ADostovalova\\Desktop\\work\\deepgmm\\tests\\test_dgmm_funcs.R  C:\\Users\\ADostovalova\\PycharmProjects\\test2\\train_data.csv')

'''
def create_encoder(representation_dim, input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(representation_dim, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)





class RepresentationLearner(keras.Model):
    def __init__(
        self,
        encoder,
        projection_units,
        num_augmentations,
        temperature=1.0,
        dropout_rate=0.1,
        l2_normalize=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        # Create projection head.
        self.projector = keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(units=projection_units, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = keras.ops.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = keras.utils.normalize(feature_vectors)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
            tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
            / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = keras.ops.max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = keras.ops.tile(
            tf.eye(batch_size), [num_augmentations, num_augmentations]
        )
        # Compute cross entropy loss
        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        # Preprocess the input images.
        preprocessed = data_preprocessing(inputs)
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}



encoder = create_encoder(representation_dim)
# Create representation learner.
representation_learner = RepresentationLearner(
    encoder, projection_units, num_augmentations=2, temperature=0.1
)
# Create a a Cosine decay learning rate scheduler.
lr_scheduler = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=500, alpha=0.1
)
# Compile the model.
representation_learner.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
    jit_compile=False,
)
# Fit the model.
history = representation_learner.fit(
    x=x_data,
    batch_size=512,
    epochs=50,  # for better results, increase the number of epochs to 500.
)

batch_size = 500
# Get the feature vector representations of the images.
feature_vectors = encoder.predict(x_data, batch_size=batch_size, verbose=1)
# Normalize the feature vectores.
feature_vectors = keras.utils.normalize(feature_vectors)
'''