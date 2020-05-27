import os, sys, glob
import numpy as np
import h5py as h5

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add, Concatenate, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv3D, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

import ops
import losses
import callbacks as cb

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
             loss_weights={'grad':.5,'adversatial':5e-5, 'pixel':.5},
             output_dir='',
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
    
        # Low-resolution and high-resolution shapes
        """ DNS-Data only has one channel, when only using PS field, when using u,v,w,ps, change to 4 channels """
        self.shape_LR = (self.LR_patchsize, self.LR_patchsize, self.LR_patchsize, self.nChannels)
        self.shape_HR = (self.HR_patchsize, self.HR_patchsize, self.HR_patchsize, self.nChannels)
    
        # Learning rates
        self.lRate_G = lRate_G
        self.lRate_D = lRate_D
        
        self.sr_factor = sr_factor
        
        # Scaling of losses
        self.loss_weights = loss_weights
        
        self.output_dir = output_dir
    
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

        def upSamplingLayer(input, scale, name):
            x = Conv3D(256,data_format="channels_last", kernel_size=3, strides=1, padding='same')(input)
            x = ops.PixelShuffling(tf.shape(x), name=name, scale=2)(x)
            x = PReLU(alpha_initializer='zeros')(x)
            return x3

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
        x = upSamplingLayer(x,2, 'shuffle1')
        x = upSamplingLayer(x,2, 'shuffle2')
        
        hr_output = Conv3D(self.nChannels,data_format="channels_last", kernel_size=9, strides=1, padding='same')(x)
        model = Model(inputs=lr_input, outputs=hr_output, name='Generator')
        #model.summary()
        return model
        
    def content_loss(self, y_true,y_pred):
        """Compute the content loss: w_pixel * MSE(pixels) + w_grad * MSE(gradients)"""
        ret =  self.loss_weights['grad'] * losses.grad(y_true, y_pred)
        ret += self.loss_weights['pixel'] * losses.pixel(y_true, y_pred)
        return ret

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.content_loss,
            optimizer=Adam(self.lRate_G, 0.9,0.999),
            metrics=[losses.PSNR, losses.pixel, losses.grad]
        )
    
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
        img = Input(shape=self.shape_HR)
        
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
        Build a Residual Attention discriminator network. Not finished yet.
        """
        def interpolating(x):
            u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
            return x[0] * u + x[1] * (1 - u)

        def ResAD_output(x):
            real, fake = x
            fake_logit = (fake - K.mean(real))
            real_logit = (real - K.mean(fake))
            return [fake_logit, real_logit]

        # Input HR images
        HR_size = self.HR_patchsize
        imgs_hr      = Input(shape=(HR_size, HR_size, HR_size, self.nChannels))
        generated_hr = Input(shape=(HR_size, HR_size, HR_size, self.nChannels))

        # Create a high resolution image from the low resolution one
        real_discriminator_logits = self.discriminator(imgs_hr)
        fake_discriminator_logits = self.discriminator(generated_hr)

        total_loss = Lambda(ResAD_output, name='ResAD_output')([real_discriminator_logits, fake_discriminator_logits])
        # Output tensors to a Model must be the output of a `Layer`
        fake_logit = Lambda(lambda x: x, name='fake_logit')(total_loss[0])
        real_logit = Lambda(lambda x: x, name='real_logit')(total_loss[1])

        model = Model(inputs=[imgs_hr, generated_hr], outputs=[fake_logit, real_logit], name='ResAD')
        model.compile(optimizer=Adam(self.lRate_D), loss=losses.ResAttention)

        return model
        
    def compile_discriminator(self, model):
        """Compile the discriminator with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.lRate_D, 0.9, 0.999),
            metrics=['accuracy']
        )

    def build_GAN(self):
        """Create the combined PISRT_GAN network"""
        def GAN_output(x):
            img_hr, generated_hr = x
            # Compute the Perceptual loss based on GRADIENT-field MSE
            grad_loss = losses.grad(img_hr, generated_hr)
            
            # Compute the RaGAN loss
            fake_logit, real_logit = self.ResAD([img_hr, generated_hr])
            gen_loss = losses.ResAttention(None, [fake_logit, real_logit])

            # Compute the pixel_loss
            pixel_loss = losses.pixel(img_hr, generated_hr)
            return [grad_loss, gen_loss, pixel_loss]

        # Input LR images
        img_lr = Input(self.shape_LR)
        img_hr = Input(self.shape_HR)

        # Create a high resolution snapshot
        generated_hr = self.generator(img_lr)
        # In the combined model we only train the generator through tne GAN
        self.discriminator.trainable = False
        self.ResAD.trainable = False

        # Output tensors to a Model must be the output of a `Layer`
        total_loss = Lambda(GAN_output, name='GAN_output')([img_hr, generated_hr])
        grad_loss  = Lambda(lambda x: self.loss_weights['grad'] * x, name='grad_loss')(total_loss[0])
        gen_loss   = Lambda(lambda x: self.loss_weights['adversatial'] * x, name='gen_loss')(total_loss[1])
        pixel_loss = Lambda(lambda x: self.loss_weights['pixel'] * x, name='pixel_loss')(total_loss[2])
        loss       = Lambda(lambda x: self.loss_weights['grad']*x[0] + self.loss_weights['adversatial']*x[1] +
                                      self.loss_weights['pixel']*x[2], name='total_loss')(total_loss)
        
        # Create model
        model = Model(inputs=[img_lr, img_hr], outputs=[grad_loss, gen_loss, pixel_loss], name='GAN')
        model.add_loss(loss)
        model.add_metric(gen_loss, name='gen_loss', aggregation='mean')
        model.add_metric(pixel_loss, name='pixel_loss', aggregation='mean')
        model.add_metric(grad_loss, name='grad_loss', aggregation='mean')
        model.compile(optimizer=Adam(self.lRate_G))

        return model

    def compile_gan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=None,
            optimizer=Adam(self.lRate_G, 0.9, 0.999)
        )

    def train_generator(self, batch_size=8, step_per_epoch=4, n_epochs=100):
        """Trains the generator only"""

        self.gen_savepath = self.output_dir + '/DNS_generator.h5'
        self.log_path = self.output_dir + '/logs'
        
        if not os.path.exists(self.log_path):
             os.makedirs(self.log_path)

        callbacks = []
        
        # This callback simply writes the metrics in a summary
        logs = cb.GeneratorLogs(self.generator, self.log_path)
        callbacks.append(logs)
        
        ckpt = ModelCheckpoint(
            self.output_dir + '/SR-RRDB-G_4X.h5',
            verbose=1,
            monitor='PSNR',
            mode='max',
            save_best_only=True,
            save_weights_only=True
        )
        ckpt.set_model(self.generator)
        callbacks.append(ckpt)

        # Write the training metrics in a .csv file
        csv_logger = CSVLogger(self.output_dir + '/history_generator_only.csv', append=True, separator='\t')
        csv_logger.set_model(self.generator)
        csv_logger.on_train_begin()
        callbacks.append(csv_logger)
        
        # Grab a snapshot for the image callback
        dl = ops.DataLoader(
            self.LR_directory,
            self.HR_directory,
            batch_size=1
        )
            
        lr_image, hr_image = dl.loadRandomBatch_noise()
        
        # For each epoch, this callback will plot a slice of Low-res/Super-res/High-res channels
        cbImg = cb.ImgCallback(
            logpath   = self.log_path,
            generator = self.generator,
            lr_data   = lr_image[0:1,:,:,:,:],
            hr_data   = hr_image[0:1,:,:,:,:]
        )
        callbacks.append(cbImg)
        
        # The progress bar needs to know the metrics
        metrics_names = self.generator.metrics_names
        
        # Need to handle restart epoch... I think I will just read the csv file
        for epoch in range(n_epochs):
            dl = ops.DataLoader(self.LR_directory, self.HR_directory, batch_size=batch_size*step_per_epoch)
            lr_images, hr_images = dl.loadRandomBatch_noise()
            print("\nEpoch {}/{}".format(epoch+1,n_epochs))
            pb_i = Progbar(step_per_epoch, stateful_metrics=metrics_names, verbose=2)
            for step in range(step_per_epoch):
                lr = lr_images[step*batch_size:(step+1)*batch_size,:,:,:,:]
                hr = hr_images[step*batch_size:(step+1)*batch_size,:,:,:,:]
                logs = self.generator.train_on_batch(lr, hr)
                
                pb_val = [('loss', logs[0]),
                          ('PSNR', logs[1]),
                          ('pixel_loss', logs[2]),
                          ('grad_loss', logs[3])]

                pb_i.add(1, values=pb_val)

            cb_logs = cb.named_logs(self.generator, logs)
            for callback in callbacks:
                callback.on_epoch_end(epoch+1, cb_logs)
