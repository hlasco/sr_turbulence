import os, sys, glob
import pickle
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add, Concatenate, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv3D, Dense
from tensorflow.keras.layers import UpSampling3D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback, CSVLogger


from keras.utils.data_utils import OrderedEnqueuer
from keras.utils import conv_utils
from keras.engine import InputSpec

import ops

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import h5py as h5
import matplotlib.pyplot as plt





class PISRT_GAN():
    """
    Implementation of PIESRGAN as described in the paper
    """

    def __init__(self,
             LR_directory, HR_directory,
             LR_patchsize=16, HR_patchsize=64,
             nChannels=5,
             batch_size=4,
             lRate_G=1e-4, lRate_D=1e-7,
             sr_factor=4,
             # VGG scaled with 1/12.75 as in paper
             loss_weights={'percept':0.2,'gen':5e-5, 'pixel':1.0},
             training_mode=True,
             refer_model=None,
             ):
        """
        LR_patchsize: Size of low-resolution data
        HR_patchsize: Size of high-resolution data
        nChannels: Number of channels (u,v,w,rho,P)
        batch_size: Batch size
        sr_factor: Super-Resolution factor
        lRate_G: Learning rate of generator
        lRage_D: Learning rate of discriminator
        """
        self.LR_directory = LR_directory
        self.HR_directory = HR_directory

        # Low-resolution image dimensions
        self.LR_patchsize = LR_patchsize
        
        # High-resolution image dimensions
        self.HR_patchsize = HR_patchsize
        
        self.nChannels = nChannels
        self.batch_size = batch_size
    
        # Low-resolution and high-resolution shapes
        """ DNS-Data only has one channel, when only using PS field, when using u,v,w,ps, change to 4 channels """
        self.shape_LR = (self.LR_patchsize, self.LR_patchsize, self.LR_patchsize, self.nChannels)
        self.shape_HR = (self.HR_patchsize, self.HR_patchsize, self.HR_patchsize, self.nChannels)
    
        self.batch_shape_LR = (self.batch_size, self.LR_patchsize, self.LR_patchsize, self.LR_patchsize, self.nChannels)
        self.batch_shape_HR = (self.batch_size, self.HR_patchsize, self.HR_patchsize, self.HR_patchsize, self.nChannels)
    
        # Learning rates
        self.lRate_G = lRate_G
        self.lRate_D = lRate_D
        
        self.sr_factor = sr_factor
        
        # Scaling of losses
        self.loss_weights = loss_weights
    
        # Gan setup settings
        self.gan_loss = 'mse'
        self.dis_loss = 'binary_crossentropy'
        
        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        #self.refer_model = refer_model
        
        # If training, build rest of GAN network
        if training_mode:
            self.discriminator = self.build_discriminator()
            #self.ResAD = self.build_ResAD()
            #self.GAN = self.build_GAN()
            #self.compile_discriminator(self.ResAD)
            #self.compile_gan(self.GAN)
            self.test()

    def save_weights(self, filepath, e=None):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}generator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        self.discriminator.save_weights("{}discriminator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)


    def build_generator(self, ):
        """
        Build the generator network.
        """
        w_init = tf.random_normal_initializer(stddev=0.02)
        HR_patchsize = self.HR_patchsize
        LR_patchsize = self.LR_patchsize
        self.data_format='channels_last'

        def upSamplingLayer(x, scale):
            x = Conv3D(256,data_format="channels_last", kernel_size=3, strides=1,
                       padding='same')(x)
            x = UpSampling3D(size=2)(x)
            #x = SubpixelConv3D('upSampleSubPixel', 2)(x)
            x = PReLU(alpha_initializer='zeros')(x)
            return x

        def denseBlock(input):
            x1 = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])
    
            x2 = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])
    
            x3 = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])
    
            x4 = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  #added x3, which ESRGAN didn't include
    
            x5 = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            """here: assumed beta=0.2"""
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = denseBlock(input)
            x = denseBlock(x)
            x = denseBlock(x)
            """here: assumed beta=0.2 as well"""
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out
            

        """----------------Assembly the generator-----------------"""
        # Input low resolution image
        #shape = (LR_patchsize, LR_patchsize, LR_patchsize, self.nChannels)
        #lr_input = Input(shape=(LR_patchsize, LR_patchsize, LR_patchsize, self.nChannels))#, batch_size=self.batch_size)
        lr_input = Input(shape=self.shape_LR)
        # Pre-residual
        x_start = Conv3D(64, data_format="channels_last", kernel_size=9, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)
        # Post-residual block
        x = Conv3D(64,data_format="channels_last", kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])
        
        # Upsampling layer
        x = upSamplingLayer(x,2)
        x = upSamplingLayer(x,2)
        
        hr_output = Conv3D(self.nChannels,data_format="channels_last", kernel_size=9, strides=1, padding='same')(x)
        model = Model(inputs=lr_input, outputs=hr_output, name='Generator')
        #Uncomment this if using multi GPU model
        #model=multi_gpu_model(model,gpus=2,cpu_merge=True)
        return model
        
    def build_discriminator(self):
        """
        Build the discriminator network according to description in the paper.
        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def discriminator_block(input, filters, kernel_size, strides=1, batchNormalization=True):
            x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
            x = LeakyReLU(alpha=0.2)(x)
            if batchNormalization:
                x = BatchNormalization(momentum=0.8)(x)
            return x

        # Input high resolution image
        HR_patch = self.HR_patchsize
        img = Input(shape=self.shape_HR)#, batch_size=self.batch_size)
        
        x = discriminator_block(img,   HR_patch, 3, strides=2, batchNormalization=False)
        x = discriminator_block(x,   2*HR_patch, 3, strides=1, batchNormalization=True)
        x = discriminator_block(x,   2*HR_patch, 3, strides=2, batchNormalization=True)
        x = discriminator_block(x,   4*HR_patch, 3, strides=1, batchNormalization=True)
        x = discriminator_block(x,   4*HR_patch, 3, strides=2, batchNormalization=True)
        x = discriminator_block(x,   8*HR_patch, 3, strides=1, batchNormalization=True)
        x = discriminator_block(x,   8*HR_patch, 3, strides=2, batchNormalization=True)
        x = discriminator_block(x,  16*HR_patch, 3, strides=1, batchNormalization=True)
        x = discriminator_block(x,  16*HR_patch, 3, strides=2, batchNormalization=True)
        x = Flatten()(x)
        x = Dense(16*HR_patch)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation='sigmoid')(x)
        # Create model and compile
        model = Model(inputs=img, outputs=x, name='Discriminator')
        return model
        
    def build_ResAD(self):
        """
        Build a Residual Attention discriminator network.
        """
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        def ResAD_output(x):
            real, fake = x
            fake_logit = (fake - K.mean(real))
            real_logit = (real - K.mean(fake))
            return [fake_logit, real_logit]
            
        def ResAD_loss(y_true, y_pred):
            fake_logit = y_pred[0]
            real_logit = y_pred[1]
            ret = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                         K.binary_crossentropy(K.ones_like(real_logit), real_logit))
            return ret

        # Input HR images
        HR_size = self.HR_patchsize
        imgs_hr      = Input(shape=(HR_size, HR_size, HR_size, self.nChannels))
        generated_hr = Input(shape=(HR_size, HR_size, HR_size, self.nChannels))
        
        # Create a high resolution image from the low resolution one
        real_discriminator_logits = self.discriminator(imgs_hr)
        fake_discriminator_logits = self.discriminator(generated_hr)
        
        total_loss = Lambda(ResAD_output, name='ResAD_output')([real_discriminator_logits, fake_discriminator_logits])
        # Output tensors to a Model must be the output of a Keras `Layer`
        fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
        real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])
        
        dis_loss = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                          K.binary_crossentropy(K.ones_like(real_logit), real_logit))

        model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit], name='ResAD')
        model.compile(optimizer=Adam(self.lRate_D), loss=ResAD_loss)

        return model


    def build_GAN(self):
        """Create the combined PISRT_GAN network"""
        def GAN_output(x):
            img_hr, generated_hr = x
           # Compute the Perceptual loss ###based on GRADIENT-field MSE
            grad_hr_x = ops.grad_x(img_hr, nChannels=self.nChannels)
            grad_hr_y = ops.grad_y(img_hr, nChannels=self.nChannels)
            grad_hr_z = ops.grad_z(img_hr, nChannels=self.nChannels)
            grad_sr_x = ops.grad_x(generated_hr, nChannels=self.nChannels)
            grad_sr_y = ops.grad_y(generated_hr, nChannels=self.nChannels)
            grad_sr_z = ops.grad_z(generated_hr, nChannels=self.nChannels)

            grad_diff = 1./3*K.mean(K.square(grad_hr_x-grad_sr_x)) + \
                        1./3*K.mean(K.square(grad_hr_y-grad_sr_y)) + \
                        1./3*K.mean(K.square(grad_hr_z-grad_sr_z))
            grad_norm = 1./3*K.mean(K.square(grad_hr_x)) + \
                        1./3*K.mean(K.square(grad_hr_y)) + \
                        1./3*K.mean(K.square(grad_hr_z))
            grad_loss = grad_diff / grad_norm
            
            # Compute the RaGAN loss
            fake_logit, real_logit = self.ResAD([img_hr, generated_hr])
            gen_loss = K.mean(
                K.binary_crossentropy(K.zeros_like(real_logit), real_logit) +
                K.binary_crossentropy(K.ones_like(fake_logit), fake_logit))

            # Compute the pixel_loss with L1 loss
            pixel_diff = K.mean(K.square(generated_hr-img_hr))
            pixel_norm = K.mean(K.square(img_hr))
            pixel_loss = pixel_diff / pixel_norm
            return [grad_loss, gen_loss, pixel_loss]

        # Input LR images
        img_lr = Input(self.shape_LR)#, batch_size=self.batch_size)
        img_hr = Input(self.shape_HR)#, batch_size=self.batch_size)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        # In the combined model we only train the generator
        self.discriminator.trainable = False
        self.ResAD.trainable = False
        # Output tensors to a Model must be the output of a Keras `Layer`

        total_loss = Lambda(GAN_output, name='GAN_output')([img_hr, generated_hr])
        grad_loss = Lambda(lambda x: self.loss_weights['percept'] * x, name='grad_loss')(total_loss[0])
        gen_loss = Lambda(lambda x: self.loss_weights['gen'] * x, name='gen_loss')(total_loss[1])
        pixel_loss = Lambda(lambda x: self.loss_weights['pixel'] * x, name='pixel_loss')(total_loss[2])
        loss = Lambda(lambda x: self.loss_weights['percept']*x[0]+
                                self.loss_weights['gen']*x[1]+
                                self.loss_weights['pixel']*x[2], name='total_loss')(total_loss)
        
        # Create model
        model = Model(inputs=[img_lr, img_hr], outputs=[grad_loss, gen_loss, pixel_loss], name='GAN')
        model.add_loss(loss)
        model.add_metric(gen_loss, name='gen_loss', aggregation='mean')
        model.add_metric(pixel_loss, name='pixel_loss', aggregation='mean')
        model.add_metric(grad_loss, name='grad_loss', aggregation='mean')
        model.compile(optimizer=Adam(self.lRate_G))

        return model

    def pixel_loss(self, y_true, y_pred):
        diff = K.mean(K.square(y_true-y_pred))
        norm = K.mean(K.square(y_true))
        loss = diff / norm
        return loss
    def mae_loss(self, y_true, y_pred):
        diff = K.mean(K.sqrt(K.square(y_true-y_pred)))
        norm = K.mean(K.sqrt(K.square(y_true)))
        loss = diff / norm
        return loss
    def grad_loss(self, y_true,y_pred):
        grad_hr_x = ops.grad_x(y_true, nChannels=self.nChannels)
        grad_hr_y = ops.grad_y(y_true, nChannels=self.nChannels)
        grad_hr_z = ops.grad_z(y_true, nChannels=self.nChannels)
        grad_sr_x = ops.grad_x(y_pred, nChannels=self.nChannels)
        grad_sr_y = ops.grad_y(y_pred, nChannels=self.nChannels)
        grad_sr_z = ops.grad_z(y_pred, nChannels=self.nChannels)
    
        grad_diff = 1./3*K.mean(K.square(grad_hr_x-grad_sr_x)) + \
                    1./3*K.mean(K.square(grad_hr_y-grad_sr_y)) + \
                    1./3*K.mean(K.square(grad_hr_z-grad_sr_z))
        grad_norm = 1./3*K.mean(K.square(grad_hr_x)) + \
                    1./3*K.mean(K.square(grad_hr_y)) + \
                    1./3*K.mean(K.square(grad_hr_z))
        grad_loss = grad_diff / grad_norm
        return grad_loss
    def loss(self, y_true,y_pred):
        ret =  self.loss_weights['percept'] * self.grad_loss(y_true, y_pred)
        ret += self.loss_weights['pixel'] * self.pixel_loss(y_true, y_pred)
        return ret
        
    def PSNR(self, y_true, y_pred):
        """
        Peek Signal to Noise Ratio
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.loss,
            optimizer=Adam(self.lRate_G, 0.9,0.999),
            metrics=['mse','mae', self.PSNR, self.pixel_loss, self.grad_loss]
        )

    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.lRate_D, 0.9, 0.999),
            metrics=['accuracy']
        )

    def compile_gan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.lRate_G, 0.9, 0.999)
        )

    def test(self,
             refer_model=None,
             batch_size=1,
             boxsize=128,
             output_name=None):
        """Trains the generator part of the network"""

        self.gen_savepath = './model/DNS_generator.h5'
        now = datetime.datetime.now()
        self.log_path = './model/logs_{}_{}_{}_{}'.format(now.year, now.month, now.day, now.hour)
        if not os.path.exists(self.log_path):
             os.makedirs(self.log_path)

        ckpts = sorted(glob.glob('./model/checkpoint/SR-RRDB-G_*.h5'))
        if os.path.exists(self.gen_savepath):
            self.generator = tf.keras.models.load_model(self.gen_savepath, 
                                custom_objects={'loss': self.loss, 'pixel_loss': self.pixel_loss, 'PSNR': self.PSNR,
                                                'grad_loss': self.grad_loss, 'mae_loss': self.mae_loss,})
        if ckpts:
            self.gen_epoch = int(ckpts[-1].split('_')[-1][:-3])
        else:
            self.gen_epoch = 0

        callbacks = []
        tensorboard = TensorBoard(
            log_dir=self.log_path,
            histogram_freq=0,
            update_freq="batch",
            write_graph=False,
            write_grads=False,
        )
        tensorboard.set_model(self.generator)
        callbacks.append(tensorboard)

        modelcheckpoint = ModelCheckpoint(
            './model/checkpoint/SR-RRDB-G_4X.h5',
            verbose=1,
            monitor='PSNR',
            mode='max',
            save_best_only=True,
            save_weights_only=True
        )
        callbacks.append(modelcheckpoint)
        csv_logger = CSVLogger('./model/history_log.csv', append=True)
        callbacks.append(csv_logger)
        
        # Create data loaders
        dl = ops.DataLoader(self.LR_directory, self.HR_directory, batch_size=1)
        lr_image, hr_image = dl.loadRandomBatch()
        
        cbImg = ops.ImgCallback( self.log_path, self.generator, 
            lr_data=lr_image[0:1,:,:,:,:], hr_data=hr_image[0:1,:,:,:,:])
        callbacks.append(cbImg)
        

        for batch_no in range(100):
            dl = ops.DataLoader(self.LR_directory, self.HR_directory, batch_size=32)
            lr_image, hr_image = dl.loadRandomBatch()
            self.generator.fit(lr_image, hr_image, batch_size=4, epochs=10*(1+batch_no) + self.gen_epoch, 
                                initial_epoch=10*batch_no + self.gen_epoch, callbacks=callbacks, verbose=2)
            self.generator.save(self.gen_savepath)
       #sr_output = self.generator.predict(lr_image,steps=1)
       #box_sr = sr_output[0,:,:,:,0]
       #box_lr = lr_image[0,:,:,:,0]
       #box_hr = hr_image[0,:,:,:,0]
       ##create image slice for visualization
       #img_hr = box_hr[:,:,0]
       #img_lr = box_lr[:,:,0]
       #img_sr = box_sr[:,:,0]

       #print(">> Ploting test images")
       #ops.Image_generator(img_lr,img_sr,img_hr)