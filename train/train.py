import time
import os
import tensorflow as tf

from train.models.MnasNet_models import Build_MnasNet
from tensorflow.keras.optimizers import SGD
from preprocess.format import PProcess
from util.util import TYPE, get_save_loc, Models
from train.models.ResNet_v2_2DCNN import ResNetv2
import visualkeras


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif 3 <= epoch < 7:
        return 1e-4
    else:
        return 1e-5


class Train:
    def __init__(self, batch_size, size, name, epochs=100):
        # # Hyper Parameters
        # self.input_dimension = 11
        self.learning_rate = 0.00001
        self.momentum = 0.85
        # self.hidden_initializer = random_uniform(seed=123)
        self.dropout_rate = 0.2

        # Configurations
        self.model_name = name
        self.model_width = 16
        self.num_channel = 1
        self.problem_type = 'Classification'
        self.output_nums = 2
        self.batch_size = batch_size
        self.tensor_width = size
        self.save_location = get_save_loc(self.model_name)
        self.epochs = epochs
        # Start working
        self.processed = PProcess(self.batch_size, self.model_name, self.tensor_width)
        self.processed.preprocess()
        self.model = None

    def build_model(self):
        """
        Used to define model and optimizer.
        """

        # Generators
        training_generator = tf.data.Dataset.from_generator(
            generator=lambda: self.processed.generator(TYPE.train, self.model_name),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [self.batch_size, self.tensor_width, self.tensor_width] if self.model_name == "raw" else [self.batch_size, self.tensor_width, 1],
                [None, 2]
            )
        )
        validation_generator = tf.data.Dataset.from_generator(
            generator=lambda: self.processed.generator(TYPE.validation, self.model_name),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [self.batch_size, self.tensor_width, self.tensor_width] if self.model_name == "raw" else [self.batch_size, self.tensor_width, 1],
                [None, 2]
            )
        )

        # Model
        if self.model_name == "raw":
            model = ResNetv2(self.tensor_width, self.tensor_width, self.num_channel, self.model_width, problem_type=self.problem_type, output_nums=self.output_nums, pooling='max', dropout_rate=self.dropout_rate).ResNet18()
        else:
            # model = Build_MnasNet('a1', dict(input_shape=(self.tensor_width, 1, 1), dropout_rate=self.dropout_rate, normalize_input=False, num_classes=2))
            ResNetv2(self.tensor_width, 1, self.num_channel, self.model_width,
                     problem_type=self.problem_type, output_nums=self.output_nums, pooling='max',
                     dropout_rate=self.dropout_rate).ResNet18()

        # Optimizer
        sgd = SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        checkpoint_prefix = os.path.join(self.save_location + "checkpoints", "ckpt_{epoch}")
        callbacks = [
            tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=self.save_location + "backup"),
            tf.keras.callbacks.TensorBoard(log_dir=self.save_location + f"logs/{self.model_name}-{int(time.time())}", write_graph=True, write_steps_per_second=True, histogram_freq=1, update_freq='batch'),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
            tf.keras.callbacks.LearningRateScheduler(decay),
        ]
        return model, sgd, callbacks, training_generator, validation_generator

    def train(self):
        """
        Main training function. Takes a training dataframe and exports a saved model.
        """
        # Distributed Strategy
        tf.distribute.MirroredStrategy()

        model, sgd, cb, training_generator, validation_generator = self.build_model()

        visualkeras.layered_view(model, legend=True, to_file=self.save_location + 'model.png')

        model.compile(
            loss='mean_squared_error',
            optimizer=sgd,
            metrics=['accuracy']
        )

        model.fit(
            steps_per_epoch=len(self.processed.data[TYPE.train].index) // self.batch_size,
            validation_steps=len(self.processed.data[TYPE.validation].index) // self.batch_size,
            use_multiprocessing=True,
            workers=6,
            x=training_generator,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=cb
        )

        model.save(self.save_location + 'saved-model/')

        self.model = model
