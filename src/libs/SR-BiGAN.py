#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This is the script for SR-BiGAN
## codes are following the structure of https://github.com/MathiasGruber/SRGAN-Keras
# step 0. load packages
import os, cv2
import sys
import pickle
import datetime
import numpy as np

## import keras + tensorflow
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Add, Reshape, Flatten
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense
from keras.layers import UpSampling2D, Lambda, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

## import sequence
import gc
from PIL import Image
from random import choice
from keras.utils import Sequence

# step 1. define data loader
def load_batch(path_lr, path_hr, batch_size, idx_vec, idx_cur=0):
    """
    : path for high-resolution images
    : path for low-resolution images
    : batch size
    : index vector
    : current index
    """
    ## get file names 
    filenames = os.listdir(path_hr)
    total_imgs = len(filenames)
    ## get images
    imgs_lr, imgs_hr = [], []
    for i in range(batch_size):
        img_name = filenames[idx_vec[idx_cur]]
        img_lr = cv2.imread(os.path.join(path_lr, img_name))
        img_hr = cv2.imread(os.path.join(path_hr, img_name))
        img_lr = img_lr/127.5 - 1
        img_hr = img_hr/127.5 - 1
        imgs_lr.append(img_lr)
        imgs_hr.append(img_hr)
    imgs_lr = np.array(imgs_lr)
    imgs_hr = np.array(imgs_hr)
    return imgs_lr, imgs_hr

# step 2. define SR-BiGAN
class SRBiGAN():
    def __init__(self, 
        height_lr=25, width_lr=25, channels=3,
        upscaling_factor=4, 
        gen_lr=1e-4, dis_lr=1e-4, 
        # VGG scaled with 1/12.75 as in paper
        loss_weights=[1e-3, 0.006], 
        training_mode=True
    ):
        """        
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """
        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError('Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        
        # Scaling of losses
        self.loss_weights = loss_weights

        # Gan setup settings
        self.gan_loss = 'mse'
        self.dis_loss = 'binary_crossentropy'
        
        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        # If training, build rest of GAN network
        if training_mode:
            self.vgg = self.build_vgg()
            self.compile_vgg(self.vgg)
            self.discriminator = self.build_discriminator()
            self.compile_discriminator(self.discriminator)
            self.encoder = self.build_encoder()
            self.compile_encoder(self.encoder)
            self.srbigan_encoder = self.build_srbigan_encoder()
            self.compile_srbigan_encoder(self.srbigan_encoder)
            self.srbigan_generator = self.build_srbigan_generator()
            self.compile_srbigan_generator(self.srbigan_generator)
            
            
    def save_weights(self, filepath):
        """Save the generator, the encoder and discriminator networks"""
        self.generator.save_weights("{}_generator_{}X.h5".format(filepath, self.upscaling_factor))
        self.encoder.save_weights("{}_encoder_{}X.h5".format(filepath, self.upscaling_factor))
        self.discriminator.save_weights("{}_discriminator_{}X.h5".format(filepath, self.upscaling_factor))

    def load_weights(self, generator_weights=None, encoder_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if encoder_weights:
            self.encoder.load_weights(encoder_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)
            
    def SubpixelConv2D(self, name, scale=2):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space
        
        :param scale: upsampling scale compared to input_shape. Default=2
        :return:
        """

        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        def subpixel(x):
            return tf.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)

    def build_vgg(self):
        """
        Load pre-trained VGG weights from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        """

        # Input image to extract features from
        img = Input(shape=self.shape_hr)

        # Get the vgg network. Extract features from last conv layer
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[20].output]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        return model  
    
    def preprocess_vgg(self, x):
        """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
        if isinstance(x, np.ndarray):
            return preprocess_input((x+1)*127.5)
        else:            
            return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)     
        
    def build_generator(self, residual_blocks=4):
        """
        Build the generator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """

        def residual_block(input):
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x = BatchNormalization(momentum=0.8)(x)
            x = PReLU(shared_axes=[1,2])(x)            
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, input])
            return x

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_'+str(number))(x)
            x = self.SubpixelConv2D('upSampleSubPixel_'+str(number), 2)(x)
            x = PReLU(shared_axes=[1,2], name='upSamplePReLU_'+str(number))(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
        x_start = PReLU(shared_axes=[1,2])(x_start)

        # Residual blocks
        r = residual_block(x_start)
        for _ in range(residual_blocks - 1):
            r = residual_block(r)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, x_start])
        
        # Upsampling depending on factor
        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)
        
        # Generate high resolution output
        # tanh activation, see: 
        # https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
        hr_output = Conv2D(
            self.channels, 
            kernel_size=9, 
            strides=1, 
            padding='same', 
            activation='tanh'
        )(x)
        
        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)        
        return model   
    
    def build_encoder(self):
        img = Input(shape=(100,100,3))
        ## layer 1: regular layer
        x = Conv2D(32, 5, strides=1, padding='same')(img)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.3)(x)

        ## layer 2: down-sampling layer
        x = Conv2D(16, 5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.3)(x)

        ## layer 3: down-sampling layer
        x = Conv2D(8, 5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.3)(x)

        ## layer 4: regular layer
        x = Conv2D(3, 5, strides=1, padding='same', activation='tanh')(x)
        model = Model(inputs=img, outputs=x)
        return model

    def build_discriminator(self, filters=16):
        """
        Build the discriminator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input high resolution image
        #input_discriminator_shape = self.shape_hr
        #input_discriminator_shape[0] = input_discriminator_shape[0]*2
        img = Input(shape=(100,100,3))
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters*2)
        x = conv2d_block(x, filters*2, strides=2)
        x = conv2d_block(x, filters*4)
        x = conv2d_block(x, filters*4, strides=2)
        x = conv2d_block(x, filters*8)
        x = conv2d_block(x, filters*8, strides=2)
        x = Dense(filters*16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        # Create model and compile
        model = Model(inputs=img, outputs=x)
        return model
    
    
    def compile_vgg(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss='mse',
            optimizer=Adam(0.0001, 0.9),
            metrics=['accuracy']
        )

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=['mse', self.PSNR]
        )
    
    def compile_encoder(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=['mse', self.PSNR]
        )
        
    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.dis_loss,
            optimizer=Adam(self.dis_lr, 0.9),
            metrics=['accuracy']
        )
    
    def compile_srbigan_generator(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=[self.dis_loss, self.gan_loss],
            loss_weights=self.loss_weights,
            optimizer=Adam(self.gen_lr, 0.9)
        )
        
    def compile_srbigan_encoder(self, model):
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9)
        )
    
    def PSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        
        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 
    def build_srbigan_generator(self):
        # Input LR images
        img_lr = Input(self.shape_lr)
        
        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        generated_features = self.vgg(
            self.preprocess_vgg(generated_hr)
        )
        
        
        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator HR images are OK
        
        generated_check = self.discriminator(generated_hr)
        
        # Create sensible names for outputs in logs
        generated_features = Lambda(lambda x: x, name='Content')(generated_features)
        generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)

        model = Model(inputs=img_lr, outputs=[generated_check, generated_features])        
        return model
    
    def build_srbigan_encoder(self):
        # Input HR images
        img_hr = Input(self.shape_hr)
        
        # Create a high resolution image from the low resolution one
        generated_lr = self.encoder(img_hr)
        
        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator LR images are OK
        
        generated_check = self.discriminator(generated_lr)
        
        model = Model(inputs=img_hr, outputs=generated_check)        
        return model
    
    def train_srbigan(self, 
        epochs, batch_size, 
        dataname, 
        path_lr,
        path_hr,
        print_frequency=1,
        log_weight_frequency=None, 
        log_weight_path='data/weights/'       
    ):
        """Train the SRGAN network
      
        """

        # Shape of output from discriminator
        disciminator_output_shape = list(self.discriminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        # VALID / FAKE targets for discriminator
        real = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape)        

        # Each epoch == "update iteration" as defined in the paper        
        print_losses = {"G": [], "E":[], "D": []}
        start_epoch = datetime.datetime.now()
        idx_vec = np.random.choice(np.arange(10000), 10000, replace=False)
        # Loop through epochs / iterations
        for epoch in range(0, epochs):
            if epoch % 100 == 0:
                idx_vec = np.random.choice(np.arange(10000), 10000, replace=False)

            # Start epoch time
            if epoch % (print_frequency + 1) == 0:
                start_epoch = datetime.datetime.now()            
            # Train discriminator
            ## combine images
            idx_cur = epoch * batch_size % 10000
            imgs_lr, imgs_hr = load_batch(path_lr, path_hr, batch_size, idx_vec, idx_cur)
            ## upscale real images
            imgs_ups = np.repeat(imgs_lr, self.upscaling_factor, axis=1)
            imgs_ups = np.repeat(imgs_ups, self.upscaling_factor, axis=2)
            generated_hr = self.generator.predict(imgs_lr)
            generated_lr = self.encoder.predict(imgs_hr)
            
            if epoch % 30 == 0:
                real_loss_1 = self.discriminator.train_on_batch(imgs_ups, real)
                fake_loss_1 = self.discriminator.train_on_batch(generated_hr, fake)
                real_loss_2 = self.discriminator.train_on_batch(generated_lr, real)
                fake_loss_2 = self.discriminator.train_on_batch(imgs_hr, fake)
                discriminator_loss = 0.5 * np.sum([real_loss_1, fake_loss_1, real_loss_2, fake_loss_2])

            # Train generator and encoder
            features_hr = self.vgg.predict(self.preprocess_vgg(imgs_hr))
            generator_loss = self.srbigan_generator.train_on_batch(imgs_lr, [real, features_hr])
            encoder_loss = self.srbigan_encoder.train_on_batch(imgs_hr, real)
            # Save losses            
            print_losses['G'].append(generator_loss)
            print_losses['E'].append(encoder_loss)
            print_losses['D'].append(discriminator_loss)

            # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['G']).mean(axis=0)
                e_avg_loss = np.array(print_losses['E']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                print('This is epoch:', epoch)

            # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:

                # Save the network weights
                self.save_weights(os.path.join(log_weight_path, dataname))


