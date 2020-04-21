#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#    
#         https://aws.amazon.com/apache-2-0/
#    
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import re

import keras
import tensorflow as tf
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop


logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)
HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
INPUT_TENSOR_NAME = 'inputs_input'  # needs to match the name of the first layer + "_input"

class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def keras_model_fn(learning_rate, weight_decay, optimizer, momentum):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a 
    TensorFlow Serving SavedModel at the end of training.

    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow 
                         training script.
    Returns: A compiled Keras model
    """

    image_size = 224

    base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.summary()
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer, loss=earth_mover_loss)

    return model


def get_filenames(channel_name, channel):
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')


def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, 'eval')


def validation_input_fn():
    return _input(args.epochs, args.batch_size, args.validation, 'validation')


def _input(epochs, batch_size, channel, channel_name):
    mode = args.data_config[channel_name]['TrainingInputMode']
    """Uses the tf.data input pipeline for CIFAR-10 dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    filenames = get_filenames(channel_name, channel)
    # Repeat infinitely.
    logging.info("Running {} in {} mode".format(channel_name, mode))
    print(filenames)
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)

    # Parse records.
    dataset = dataset.map(
        _dataset_parser, num_parallel_calls=10)

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    
    print(dataset)
    print(image_batch)
    print(label_batch)

    return {INPUT_TENSOR_NAME: image_batch}, label_batch


def _train_preprocess_fn(image):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    print('in dataset parser')
    featdef = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(value, featdef)
    image = tf.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(example['label'], tf.int32)
    #image = _train_preprocess_fn(image)
    return image, tf.one_hot(label, NUM_CLASSES)


def save_model(model, output):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': model.input}, outputs={'scores': model.output})

    builder = tf.saved_model.builder.SavedModelBuilder(output+'/1/')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        })
    builder.save()
    logging.info("Model successfully saved at: {}".format(output))
    return


def main(args):
    
    logging.info("getting data")
    
    train_dataset = train_input_fn()
    eval_dataset = eval_input_fn()
    validation_dataset = validation_input_fn()
    

    logging.info("configuring model")
    model = keras_model_fn(args.learning_rate, args.weight_decay, args.optimizer, args.momentum)

    # load weights from trained model if it exists
    if os.path.exists('weights/mobilenet_weights.h5'):
        model.load_weights('weights/mobilenet_weights.h5')

    checkpoint = ModelCheckpoint('weights/mobilenet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,mode='min')
    tensorboard = TensorBoardBatch()
    callbacks = [checkpoint, tensorboard]
    
    
    logging.info("Starting training")
    size = 1
    model.fit(x=train_dataset[0], 
              y=train_dataset[1],
              epochs=args.epochs, 
              validation_data=validation_dataset,
              callbacks=callbacks)
    

    logging.info("Starting evaluation")
    #score = model.evaluate(eval_dataset[0], eval_dataset[1], steps=num_examples_per_epoch('eval') // args.batch_size,
    #                       verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    # Horovod: Save model only on worker 0 (i.e. master)
    if mpi:
        if hvd.rank() == 0:
            save_model(model, args.model_output_dir)
    else:
        save_model(model, args.model_output_dir)


def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 40000
    elif subset == 'validation':
        return 10000
    elif subset == 'eval':
        return 10000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=21,
        help='Batch size for training.')
    parser.add_argument(
        '--data-config',
        type=json.loads,
        default=os.environ.get('SM_INPUT_DATA_CONFIG')
    )
    parser.add_argument(
        '--fw-params',
        type=json.loads,
        default=os.environ.get('SM_FRAMEWORK_PARAMS')
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default='0.9'
    )
    args = parser.parse_args()
    main(args)