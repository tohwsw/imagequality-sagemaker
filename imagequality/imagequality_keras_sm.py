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
import numpy as np
import glob


import keras
import tensorflow as tf
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop

from datetime import datetime
#from data_loader import train_generator, val_generator


logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)
IMAGE_SIZE = 224
HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
INPUT_TENSOR_NAME = 'input_1_input'  # needs to match the name of the first layer + "_input"


class Sync2S3(tf.keras.callbacks.Callback):
    def __init__(self, logdir, s3logdir):
        super(Sync2S3, self).__init__()
        self.logdir = logdir
        self.s3logdir = s3logdir
        #print('s3logdir',s3logdir)

    def on_epoch_end(self, batch, logs={}):
        os.system('aws s3 sync '+self.logdir+' '+self.s3logdir)
        # ' >/dev/null 2>&1'


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
            writer = tf.summary.FileWriter(log_dir, sess.graph)
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

    base_model = MobileNet((IMAGE_SIZE, IMAGE_SIZE, 3), alpha=1, include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x = Dropout(0.75)(base_model.output)
    
    # follow the number of classes
    x = Dense(7, activation='sigmoid')(x)
    #x = Dense(1, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.summary()
    optimizer = Adam(lr=learning_rate)
    # will need to change the loss function to binary cross entropy
    model.compile(optimizer, loss=earth_mover_loss)

    return model


def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')


def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, 'eval')


def validation_input_fn():
    return _input(args.epochs, args.batch_size, args.validation, 'validation')


def _input(epochs, batch_size, channel, channel_name):
    # Read the imges into memory and divide the images into training and validation
    # path to the images and the text file which holds the scores and ids
    base_images_path = '/opt/ml/input/data/train/'
    #ava_dataset_path = '/opt/ml/input/data/eval/AVA.txt'
    ava_dataset_path = '/opt/ml/input/data/train/labels.txt'

    files = glob.glob(base_images_path + "*.jpg")
    files = sorted(files)

    train_image_paths = []
    train_scores = []

    print("Loading training set and val set")
    with open(ava_dataset_path, mode='r') as f:
        lines = f.readlines()
        print('lines',lines)
        for i, line in enumerate(lines):
            token = line.split()
            
            # 2 to 12 are the scores
            #values = np.array(token[2:12], dtype='float32')
            values = np.array(token[2:2], dtype='float32')
            values /= values.sum()
            #print('values',values)

            file_path = base_images_path + str(token[1]) + '.jpg'
            #print('file_path',file_path)
            if os.path.exists(file_path):
                train_image_paths.append(file_path)
                train_scores.append(values)
            
            # 255000 samples divided by 20 epoches
            count = 255000 // 20 
            if i % count == 0 and i != 0:
                print('Loaded %d percent of the dataset' % (i / 255000. * 100))

    train_image_paths = np.array(train_image_paths)
    train_scores = np.array(train_scores, dtype='float32')

    #val_image_paths = train_image_paths[-5000:]
    #val_scores = train_scores[-5000:]
    #train_image_paths = train_image_paths[:-5000]
    #train_scores = train_scores[:-5000]
    val_image_paths = train_image_paths
    val_scores = train_scores
    
    print('Train set size : ', train_image_paths.shape, train_scores.shape)
    print('Val set size : ', val_image_paths.shape, val_scores.shape)
    print('Train and validation datasets ready !')
    
    return train_image_paths, train_scores, val_image_paths, val_scores

def train_generator(batchsize,train_image_paths,train_scores):
    '''
    Creates a python generator that loads the AVA dataset images with random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    print('========= In Train Generator ==========')
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)
        
        # Combines consecutive elements of this dataset into batches.
        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        shuffle=True
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)

def val_generator(batchsize,val_image_paths,val_scores):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for validation set

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_scores))
        val_dataset = val_dataset.map(parse_data_without_augmentation)

        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)

def parse_data(filename, scores):
    '''
    Loads the image file, and randomly applies crops and flips to each image.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    print('=========== In Parse Data ============')
    print('filename',filename)
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (256, 256))
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

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

def serving_input_fn(hyperparameters):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main(args):
    
    logging.info("getting data")
    
    train_dataset = train_input_fn()
    
    
    #eval_dataset = eval_input_fn()
    #validation_dataset = validation_input_fn()
    
    logging.info("configuring model")
    model = keras_model_fn(args.learning_rate, args.weight_decay, args.optimizer, args.momentum)

    # load weights from trained model if it exists
    #if os.path.exists('mobilenet_weights.h5'):
    #    print('Loaded weights from trained model')
    #    model.load_weights('mobilenet_weights.h5')

    checkpoint = ModelCheckpoint('mobilenet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,mode='min')
    #tensorboard = TensorBoardBatch()
    
    callbacks = [checkpoint]
    
    
    logging.info("Starting training")

    #epochs = 20
    
    model.fit_generator(train_generator(args.batch_size,train_dataset[0],train_dataset[1]),
                    #steps_per_epoch=(250000. // batchsize),
                    steps_per_epoch=(25000. // args.batch_size),
                    epochs=args.epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_generator(args.batch_size,train_dataset[2],train_dataset[3]),
                    validation_steps=(5000. // args.batch_size))
    

    logging.info("Starting evaluation")
    #score = model.evaluate(eval_dataset[0], eval_dataset[1], steps=num_examples_per_epoch('eval') // args.batch_size,
    #                       verbose=0)

    #logging.info('Test loss:{}'.format(score[0]))
    #logging.info('Test accuracy:{}'.format(score[1]))

    # Horovod: Save model only on worker 0 (i.e. master)

    save_model(model, args.model_output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where the NIMA input data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The directory where the NIMA input data is stored.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='The directory where the NIMA input data is stored.')
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