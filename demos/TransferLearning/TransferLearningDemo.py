import argparse
import os
import warnings

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import optimizers

warnings.simplefilter('ignore')


class Demo:
    # Cats & Dogs classes
    NUM_CLASSES = 2

    # RGB
    CHANNELS = 3

    RESNET50_POOLING_AVERAGE = 'avg'
    DENSE_LAYER_ACTIVATION = 'softmax'
    OBJECTIVE_FUNCTION = 'categorical_crossentropy'

    # Common accuracy metric for all outputs, but can use different metrics for different output
    LOSS_METRICS = ['accuracy']

    SGD = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    IMAGE_SIZE = 224
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    def __init__(self,
                 training=0,
                 warmup=0,
                 predict=0,
                 epochs=5,
                 batch_size=16,
                 model_name='ResNet50',
                 backend='CPU',
                 workers=1,
                 verbose=1,
                 callbacks=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose

        # Images
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.test_class_indices = []
        self.init_images()

        self.setup_ngraph_bridge(backend=backend)

        self.compile_model(model_name)

        if training:
            if warmup:
                self.train(epochs=1, callbacks=callbacks)
            h = self.train(epochs=self.epochs, callbacks=callbacks)
        if predict:
            p = self.predict(callbacks)

    def init_images(self):
        zip_file = tf.keras.utils.get_file(
            origin=
            "https://github.com/plaidml/depot/raw/master/datasets/cats_and_dogs_filtered.zip",
            fname="cats_and_dogs_filtered.zip",
            extract=True)
        base_dir = os.path.dirname(zip_file)
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')
        self.test_dir = os.path.join(base_dir, 'test')

        # Directory with our training cat pictures
        train_cats_dir = os.path.join(train_dir, 'cats')

        # Directory with our training dog pictures
        train_dogs_dir = os.path.join(train_dir, 'dogs')

        # Directory with our validation cat pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')

        # Directory with our validation dog pictures
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')

        # Directory with our test cat pictures
        test_cats_dir = os.path.join(self.test_dir, 'cats')

        # Directory with our test dog pictures
        test_dogs_dir = os.path.join(self.test_dir, 'dogs')

        # Preprocess images
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        #with self.out_stats:
        # Flow training images in batches of 20 using train_datagen generator
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,  # Source directory for the training images
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.batch_size,
            class_mode='categorical')

        # Flow validation images in batches of 20 using test_datagen generator
        self.validation_generator = validation_datagen.flow_from_directory(
            validation_dir,  # Source directory for the validation images
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.batch_size,
            class_mode='categorical')

        # Flow validation images in batches of 20 using test_datagen generator
        self.test_generator = validation_datagen.flow_from_directory(
            self.test_dir,  # Source directory for the test images
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False,
            seed=42)

        # Test Correct Values (0 Cat, 1 Dog)
        for file in self.test_generator.filenames:
            if "cat" in file:
                self.test_class_indices.append(0)
            elif "dog" in file:
                self.test_class_indices.append(1)
            else:
                print("Error, unclassifiable image " + file)

    def setup_ngraph_bridge(self, backend):
        # Enviornment variables
        os.environ['PLAIDML_USE_STRIPE'] = '1'

        if self.workers < 1:
            os.environ['OMP_NUM_THREADS'] = 1
        else:
            # Use default
            if os.getenv('OMP_NUM_THREADS') is not None:
                del os.environ['OMP_NUM_THREADS']

        import ngraph_bridge

        if backend == 'DISABLED' or backend == 'TF':
            ngraph_bridge.disable()
        elif backend == 'CPU':
            ngraph_bridge.set_backend('CPU')
            ngraph_bridge.enable()
        elif backend == 'PLAIDML':
            ngraph_bridge.set_backend('PLAIDML')
            ngraph_bridge.enable()
        else:
            print("ERROR: Unsupported backend " + backend + " selected.")

    def compile_model(self, model_name, fine=0):
        if model_name == 'ResNet50':
            self.base_model = ResNet50(pooling=self.RESNET50_POOLING_AVERAGE,
                                       include_top=False,
                                       weights='imagenet')
            self.model = tf.keras.Sequential([
                self.base_model,
                keras.layers.Dense(self.NUM_CLASSES, activation=self.DENSE_LAYER_ACTIVATION)
            ])
        elif model_name == 'MobileNet v2':
            self.base_model = MobileNetV2(input_shape=self.IMAGE_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
            self.model = tf.keras.Sequential([
                self.base_model,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(self.NUM_CLASSES, activation=self.DENSE_LAYER_ACTIVATION)
            ])

        # Fine Tuning
        if fine:
            self.base_model.trainable = True
            # Fine tune from this layer onwards
            self.fine_tune_at = fine

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_model.layers[:self.fine_tune_at]:
                layer.trainable = False

        else:
            self.base_model.trainable = False

        self.model.compile(optimizer=self.SGD,
                           loss=self.OBJECTIVE_FUNCTION,
                           metrics=self.LOSS_METRICS)

    def train(self, epochs, callbacks=None):
        steps_per_epoch = self.train_generator.n // self.batch_size
        validation_steps = self.validation_generator.n // self.batch_size

        history = self.model.fit_generator(self.train_generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           workers=self.workers,
                                           validation_data=self.validation_generator,
                                           validation_steps=validation_steps,
                                           verbose=self.verbose,
                                           callbacks=callbacks)
        return history

    def predict(self, callbacks=None):
        probabilities = self.model.predict_generator(self.test_generator,
                                                     verbose=self.verbose,
                                                     workers=self.workers,
                                                     callbacks=callbacks)
        return probabilities


def positive_int(v):
    value = int(v)
    if value < 0:
        msg = '%r is not a positive value' % value
        raise argparse.ArgumentTypeError(msg)
    return value


def positive_nonzero_int(v):
    value = int(v)
    if value <= 0:
        msg = '%r is not a positive non-zero value' % value
        raise argparse.ArgumentTypeError(msg)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TransferLearningDemo')
    parser.add_argument('--training',
                        help='performs the training phase of the demo',
                        action='store_true')
    parser.add_argument('--predict',
                        help='performs the inference phase of the demo',
                        action='store_true')
    parser.add_argument(
        '--network_type',
        help='selects the network used for training/classification [ResNet50]/MobileNet V2',
        default='ResNet50')
    parser.add_argument(
        '--backend',
        help='selects the backend used for training/classification [CPU]/PLAIDML/TF]',
        default='PLAIDML')
    parser.add_argument('--quiet', help='disables most logging', action='store_false')
    parser.add_argument('--epochs',
                        help='number of epochs to train',
                        type=positive_nonzero_int,
                        default=5)
    parser.add_argument('--batch_size',
                        help='specify batch size for training',
                        type=positive_nonzero_int,
                        default=16)
    parser.add_argument(
        '--workers',
        help='specify number of workers for threading; zero specifies main thread only',
        type=positive_int,
        default=1)
    parser.add_argument('--warmup', help='warmup run for training', action='store_true')
    args = parser.parse_args()

    Demo(training=args.training,
         warmup=args.warmup,
         predict=args.predict,
         epochs=args.epochs,
         batch_size=args.batch_size,
         model_name=args.network_type,
         backend=args.backend,
         workers=args.workers,
         verbose=args.quiet)
