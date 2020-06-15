import argparse
import json
import logging
import re
import os


import keras
import tensorflow as tf

from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator

HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 2

def save_model(model, output):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': model.input}, outputs={'scores': model.output}
    )

    builder = tf.saved_model.builder.SavedModelBuilder(output+'/1/')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        },
    )

    builder.save()
    logging.info("Model successfully saved at: {}".format(output))


def main(args):
    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'model', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir
    logging.info("Writing TensorBoard logs to {}".format(tensorboard_dir))
    
    print("========= Creating model ========")

    # define the keras model
    base_model = MobileNet((HEIGHT, WIDTH, 3), alpha=1, include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    x = Dropout(0.75)(base_model.output)
    
    #follow the number of classes
    preds = Dense(NUM_CLASSES, activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=preds)
    model.summary

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    callbacks = []
    callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.h5'))
    callbacks.append(TensorBoard(log_dir=tensorboard_dir, update_freq='epoch'))


    print("========= Creating data generator ========")

    train_datagen = ImageDataGenerator()
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    
    train_generator = train_datagen.flow_from_directory(
            '/opt/ml/input/data/training/',  # this is the target directory
            target_size=(224, 224),  # all images will be resized to 224x224
            batch_size=args.batch_size,
            class_mode='categorical')  # For binary classification

    validation_generator = train_datagen.flow_from_directory(
            '/opt/ml/input/data/validation/',  # this is the target directory
            target_size=(224, 224),  # all images will be resized to 224x224
            batch_size=args.batch_size,
            class_mode='categorical')  # For binary classification
    
    no_of_training_images = 5000
    no_of_validation_images = 500

    print("========= Running Training ========")
    model.fit_generator(
            train_generator,
            steps_per_epoch=(no_of_training_images // args.batch_size),
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=(no_of_validation_images // args.batch_size))
    
    save_model(model, args.model_output_dir)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where theinput data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The directory where the input data is stored.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='The directory where the input data is stored.')
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
        default=128,
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

