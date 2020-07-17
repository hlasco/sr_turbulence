import os, sys, glob
import numpy as np
import h5py as h5

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add, Concatenate, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv3D, Dense, ReLU
from tensorflow.keras.layers import Lambda, UpSampling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import BinaryCrossentropy

import ops
import losses
import callbacks as cb


EPS = 1e-7

class SRISMt():
    """
    Implementation of a Physics-Informed Super Resolution GAN for Turbulent Flows
    """

    def __init__(self,
             data_directory,
             lr_patchsize=16, hr_patchsize=64,
             nChannels=4,
             batch_size=4,
             lRate_G=1e-4, lRate_D=1e-7,
             sr_factor=4,
             bNorm = False,
             loss_weights={'pixel':.7, 'TE':.1,'MF':.1,'ENS':0.1,'adversarial':5e-5, },
             output_dir='',
             training_mode=True,
             refer_model=None,
             ):
        """
        lr_patchsize: Size of low-resolution data
        hr_patchsize: Size of high-resolution data
        nChannels: Number of channels (u,v,w,rho,P)
        batch_size: Batch size
        sr_factor: Super-Resolution factor
        lRate_G: Learning rate of generator
        lRage_D: Learning rate of discriminator
        """
        self.data_directory = data_directory

        # Low-resolution image dimensions
        self.lr_patchsize = lr_patchsize

        # High-resolution image dimensions
        self.hr_patchsize = hr_patchsize

        self.nChannels = nChannels

        nGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
        print("Num GPUs Available: ", nGPUs, flush=True)


        self.strategy = tf.distribute.get_strategy()
        if nGPUs>1:
            print("Using MirroredStrategy", flush=True)
            self.strategy = tf.distribute.MirroredStrategy()
            #print("Using CentralStorageStrategy", flush=True)
            #self.strategy =tf.distribute.experimental.CentralStorageStrategy()

        # Low-resolution and high-resolution shapes
        """ DNS-Data only has one channel, when only using PS field, when using u,v,w,ps, change to 4 channels """
        self.shape_lr = (self.lr_patchsize, self.lr_patchsize, self.lr_patchsize, self.nChannels)
        self.shape_hr = (self.hr_patchsize, self.hr_patchsize, self.hr_patchsize, self.nChannels)

        # Learning rates
        self.lRate_G = lRate_G
        self.lRate_D = lRate_D

        # Super-resolution factor
        self.sr_factor = sr_factor

        # Scaling of losses
        self.loss_weights = loss_weights
        self.output_dir = output_dir
        self.bNorm = bNorm

        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        # Initial epochs
        self.epoch_0 = 0

        # If training, build and compile the combined model
        if training_mode:
            self.discriminator = self.build_discriminator()
            self.compile_discriminator(self.discriminator)
            self.discriminator.trainable = False
            self.GAN = self.build_GAN() #Build and compile happens here
            #self.discriminator.trainable = True


    def restart(self, gen_w=None, dis_w=None, epoch_0=0):
        self.epoch_0 = epoch_0
        if gen_w is not None:
            print('Restarting generator from weights: ', gen_w)
            self.generator = tf.keras.models.load_model(gen_w, compile=False)
        if dis_w is not None:
            print('Restarting discriminator from weights: ', dis_w)
            self.discriminator = tf.keras.models.load_model(dis_w, compile=False)

    def build_generator(self):
        """
        Build the generator network.
        """
        w_init = tf.random_normal_initializer(stddev=0.02)
        hr_patchsize = self.hr_patchsize
        lr_patchsize = self.lr_patchsize
        self.data_format='channels_last'

        def upSamplingLayer_shuffle(input, scale, name):
            x = Conv3D(256, data_format="channels_last", kernel_size=3, strides=1, padding='same')(input)
            x = ops.PixelShuffling(tf.shape(x), name=name, scale=2)(x)
            x = PReLU(alpha_initializer='zeros')(x)
            return x

        def upSamplingLayer(input, scale, name):
            x = UpSampling3D(size=scale, data_format="channels_last")(input)
            x = Conv3D(256, data_format="channels_last", kernel_size=3, strides=1, padding='same')(x)
            x = Activation('relu')(x)
            return x

        def RDB(inpt):
            x = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(inpt)
            x = LeakyReLU(0.2)(x)
            ret = Add()([x, inpt])
            return ret

        def RRDB(inpt, num_resblock):
            x = inpt
            for i in range(num_resblock):
                x = RDB(x)

            x = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(x)
            ret = Add()([x, inpt])
            return ret

        """----------------Assembly the generator-----------------"""
        # Input low resolution image
        #lr_input = Input(shape=self.shape_lr)
        with self.strategy.scope():

            # The generator only can deal with any input size since it's only 
            # built out of comvolution layers
            lr_input = Input(shape=(None, None, None, self.nChannels))
            
            # Pre-residual
            x_start = Conv3D(64, data_format="channels_last", kernel_size=3, strides=1, padding='same')(lr_input)
            x_start = LeakyReLU(0.2)(x_start)
    
            # My Residual Residual Dense Block
            x = RRDB(x_start, num_resblock=12)
    
            # Post-residual block
            x = Conv3D(64,data_format="channels_last", kernel_size=3, strides=1, padding='same')(x)
            if self.bNorm:
                x = BatchNormalization(momentum = 0.5)(x)
            x = Lambda(lambda x: x * 0.2)(x)
            x = Add()([x, x_start])
    
            # UpSampling3D, Conv3D(3,256,1), ReLU
            x = upSamplingLayer(x,2,'upsample1')
            x = upSamplingLayer(x,2,'upsample2')
    
            # Final layer : recombine the channels together
            hr_output = Conv3D(self.nChannels,data_format="channels_last", kernel_size=3, strides=1, padding='same')(x)
            model = Model(inputs=lr_input, outputs=hr_output, name='Generator')
        return model

    def content_loss(self, y_true,y_pred):
        """Compute the content loss"""
        ret = self.loss_weights['pixel'] * losses.pixel(y_true, y_pred)
        ret += self.loss_weights['TE'] * losses.total_energy(y_true, y_pred)
        ret += self.loss_weights['MF'] * losses.mass_flux(y_true, y_pred)
        ret += self.loss_weights['ENS'] * losses.enstrophy(y_true, y_pred)

        return ret

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        with self.strategy.scope():
            model.compile(
                loss=self.content_loss,
                optimizer=Adam(self.lRate_G, 0.9,0.999),
                metrics=[losses.PSNR, losses.pixel, losses.total_energy, losses.mass_flux, losses.enstrophy]
            )

    def build_discriminator(self):
        """
        Build the discriminator network.
        :return: model
        """

        def discriminator_block(input, filters, kernel_size, strides=1, batchNormalization=True):
            d1 = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
            d2 = LeakyReLU(alpha=0.2)(d1)
            if batchNormalization:
                d2 = BatchNormalization(momentum=0.5)(d2)
            return d2

        # Input high resolution image
        hr_patch = self.hr_patchsize//2
        lr_patch = self.lr_patchsize
        with self.strategy.scope():
            x0 = Input(shape=self.shape_hr)
            
            # Maybe batch normalization one the first layer would help
            x = discriminator_block(x0,  1*hr_patch, 3, strides=1, batchNormalization=False)
            x = discriminator_block(x,   1*hr_patch, 3, strides=2, batchNormalization=False)
            x = discriminator_block(x,   2*hr_patch, 3, strides=1, batchNormalization=False)
            x = discriminator_block(x,   2*hr_patch, 3, strides=2, batchNormalization=False)
            x = discriminator_block(x,   4*hr_patch, 3, strides=1, batchNormalization=False)
            x = discriminator_block(x,   4*hr_patch, 3, strides=2, batchNormalization=False)
            x = discriminator_block(x,   8*hr_patch, 3, strides=1, batchNormalization=False)
            x = discriminator_block(x,   8*hr_patch, 3, strides=2, batchNormalization=False)
    
            x = Flatten()(x)
            x = Dense(16*hr_patch)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.4)(x)
            validity = Dense(1, activation='sigmoid')(x)
    
            # Create model and compile
            model = Model(inputs=x0, outputs=validity, name='Discriminator')
            model.add_metric(validity, name='validity', aggregation='mean')
        return model

    def compile_discriminator(self, model):
        """Compile the discriminator with Adam optimizer"""
        with self.strategy.scope():
            model.compile(
                loss='binary_crossentropy',
                optimizer=Adam(self.lRate_D, 0.9, 0.999),
            )

    def build_GAN(self):
        """Create the combined SRISMt network"""
        def GAN_output(x):
            _img_hr, _img_sr, _adversarial = x

            # Compute the pixel_loss
            pixel = losses.pixel(_img_hr, _img_sr)

            # Compute the total_energy_loss
            total_energy = losses.total_energy(_img_hr, _img_sr)

            # Compute the mass_flux_loss
            mass_flux = losses.mass_flux(_img_hr, _img_sr)

            # Compute the enstrophy_loss
            enstrophy = losses.enstrophy(_img_hr, _img_sr)

            # Compute the enstrophy_loss
            PSNR = losses.PSNR(_img_hr, _img_sr)

            return [PSNR, _adversarial,  pixel, total_energy, mass_flux, enstrophy]

        # Input lr images
        with self.strategy.scope():
            # Size of images have to be well defined here for the discriminator
            img_lr = Input(self.shape_lr)
            img_hr = Input(self.shape_hr)
    
            # Generate the super-resolution snapshots
            img_sr = self.generator(img_lr)
    
            pred_D = self.discriminator(img_sr)
            adversarial = K.mean(-K.log(pred_D + EPS))
            #adversarial = K.mean(K.binary_crossentropy(target=tf.ones_like(pred_D), output=pred_D))
            # Output tensors to a Model must be the output of a `Layer`
            # That's a bit messy, I should be able to pack this up in a cleaner way
            gan_output   = Lambda(GAN_output, name='GAN_output')([img_hr, img_sr, adversarial])
            PSNR         = Lambda(lambda x: x, name='PSNR')(gan_output[0])
            adversarial  = Lambda(lambda x: x, name='gen')(gan_output[1])
            pixel        = Lambda(lambda x: x, name='pixel')(gan_output[2])
            total_energy = Lambda(lambda x: x, name='total_energy')(gan_output[3])
            mass_flux    = Lambda(lambda x: x, name='mass_flux')(gan_output[4])
            enstrophy    = Lambda(lambda x: x, name='enstrophy')(gan_output[5])
    
            loss   = Lambda(lambda x: self.loss_weights['adversarial']*x[1] + self.loss_weights['pixel']*x[2] +
                                      self.loss_weights['TE']*x[3] + self.loss_weights['MF']*x[4] +
                                      self.loss_weights['ENS']*x[5], name='total_loss')(gan_output)
    
            # Create model
            model = Model(inputs=[img_lr, img_hr], outputs=[PSNR, adversarial, pixel, total_energy, mass_flux, enstrophy], name='GAN')

            # Ideally I would like to add a loss while compiling the GAN but I gor errors while trying
            model.add_loss(loss)
    
            model.add_metric(PSNR, name='PSNR', aggregation='mean')
            model.add_metric(adversarial, name='adversarial', aggregation='mean')
            model.add_metric(pixel, name='pixel', aggregation='mean')
            model.add_metric(total_energy, name='total_energy', aggregation='mean')
            model.add_metric(mass_flux, name='mass_flux', aggregation='mean')
            model.add_metric(enstrophy, name='enstrophy', aggregation='mean')

            model.compile(optimizer=Adam(self.lRate_G, 0.9,0.999))
        return model

    def train_GAN(self, batch_size=8, step_per_epoch=4, n_epochs=100):
        """Trains the generator only"""
        print('Using LR_D=', self.lRate_G, ' LR_D=', self.lRate_D)
        self.log_path = self.output_dir + '/logs'

        if not os.path.exists(self.log_path):
             os.makedirs(self.log_path)

        callbacks_G = []
        callbacks_D = []

        # This callback simply writes the metrics in a summary
        logger_G = cb.GanLogs(self.GAN, self.log_path, modeltype='generator')
        callbacks_G.append(logger_G)

        # This callback simply writes the metrics in a summary
        logger_D = cb.GanLogs(self.discriminator, self.log_path, modeltype='discriminator')
        callbacks_D.append(logger_D)

        # Checkpoints for the Generator
        ckpt_G = ModelCheckpoint(
            self.output_dir + '/SR-RRDB-G_4X.h5',
            verbose=1,
            save_freq='epoch',
            save_best_only=False,
            save_weights_only=False
        )
        ckpt_G.set_model(self.generator)
        callbacks_G.append(ckpt_G)

        # Checkpoints for the Discriminator
        ckpt_D = ModelCheckpoint(
            self.output_dir + '/SR-RRDB-D_4X.h5',
            verbose=1,
            save_freq='epoch',
            save_best_only=False,
            save_weights_only=False
        )
        ckpt_D.set_model(self.discriminator)
        callbacks_D.append(ckpt_D)

        # Write the training metrics in a .csv file
        csv_logger_G = CSVLogger(self.output_dir + '/history_generator.csv', append=True, separator='\t')
        csv_logger_G.set_model(self.GAN)
        csv_logger_G.on_train_begin()
        callbacks_G.append(csv_logger_G)

        # Write the training metrics in a .csv file
        csv_logger_D = CSVLogger(self.output_dir + '/history_discriminator.csv', append=True, separator='\t')
        csv_logger_D.set_model(self.discriminator)
        csv_logger_D.on_train_begin()
        callbacks_D.append(csv_logger_D)

        # Grab a snapshot for the image callback
        dl = ops.DataLoader(
            self.data_directory,
        )

        lr_image, hr_image = dl.loadRandomBatch(batch_size=1)

        # At epoch end, this callback will plot a slice of Low-res/Super-res/High-res channels
        cbImg = cb.ImgCallback(
            logpath   = self.log_path,
            generator = self.generator,
            lr_data   = lr_image[0:1,:,:,:,:],
            hr_data   = hr_image[0:1,:,:,:,:]
        )
        callbacks_G.append(cbImg)

        # The progress bar needs to know the metric names
        metrics_names = ['loss_G','loss_D','PSNR','adv_loss','pixel_loss','energy_loss','flux_loss','enstrophy_loss']

        # Need to handle restart epoch... Atm it's a parameter declared when restarting a model
        e0 = self.epoch_0
        for epoch in range(e0, n_epochs + e0):
            lrs, hrs = dl.loadRandomBatch(batch_size=batch_size*step_per_epoch)
            print("\nEpoch {}/{}".format(epoch+1,n_epochs), flush=True)
            for step in range(step_per_epoch):

                lr = lrs[batch_size*step:batch_size*(step+1),:,:,:,:]
                hr = hrs[batch_size*step:batch_size*(step+1),:,:,:,:]

                sr = self.generator.predict(lr)

                # Here I could try to add some noise to improve the Discriminator
                labels_real = np.ones(batch_size)  - np.random.uniform(low=0.0, high=0.2, size=batch_size)
                labels_fake = np.zeros(batch_size) + np.random.uniform(low=0.0, high=0.2, size=batch_size)
                # Two training steps for the Discriminator
                logs_D_real = self.discriminator.train_on_batch(x=hr, y=labels_real)
                logs_D_fake = self.discriminator.train_on_batch(x=sr, y=labels_fake)
                # For callbacks
                loss_D = 0.5*np.add(logs_D_fake[0], logs_D_real[0])

                # Training step for the Generator
                logs_G = self.GAN.train_on_batch([lr, hr], None)
                logs_G = cb.named_logs(self.GAN, logs_G)

                print("  Step {}/{}: loss_G={:10.4f}, loss_D={:10.4f}".format(
                      step+1,step_per_epoch, logs_G['loss'], loss_D), flush=True)

                """
                dis_count = 1
                while (loss_D > 0.6) and (dis_count < 0):
                    logs_D_real = self.discriminator.train_on_batch(x=hr, y=labels_real)
                    logs_D_fake = self.discriminator.train_on_batch(x=sr, y=labels_fake)
                    loss_D = 0.5*np.add(logs_D_fake[0], logs_D_real[0])
                    print("    D substep {}, loss={:10.4f}".format(dis_count, loss_D), flush=True)
                    dis_count += 1

                adv_loss = logs_G['adversarial']
                gen_count = 1
                while (adv_loss > 0.6) and (gen_count < 0):
                    logs_G = self.GAN.train_on_batch([lr, hr], None)
                    logs_G = cb.named_logs(self.GAN, logs_G)
                    adv_loss = logs_G['adversarial']
                    loss = logs_G['loss']
                    print("    G substep {}, loss={:10.4f}, adv_loss={:10.4f}".format(gen_count, loss, adv_loss), flush=True)
                    gen_count += 1
                """

            if(epoch%10==0):
                cb_logs_G = logs_G
                for callback in callbacks_G:
                    callback.on_epoch_end(epoch+1, cb_logs_G)
                cb_logs_D = {'loss':loss_D, 'output_real':logs_D_real[1], 'output_fake':logs_D_fake[1]}
                for callback in callbacks_D:
                    callback.on_epoch_end(epoch+1, cb_logs_D)

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
            self.output_dir + '/SR-RRDB-D_4X.h5',
            verbose=1,
            save_freq='epoch',
            save_best_only=False,
            save_weights_only=False
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
            self.data_directory,
        )

        lr_image, hr_image = dl.loadRandomBatch(batch_size=1)

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
        e0 = self.epoch_0
        for epoch in range(e0, n_epochs + e0):
            lr_images, hr_images = dl.loadRandomBatch(batch_size=batch_size*step_per_epoch)
            print("\nEpoch {}/{}".format(epoch+1,n_epochs))
            pb_i = Progbar(step_per_epoch, stateful_metrics=metrics_names, verbose=2)
            for step in range(step_per_epoch):
                lr = lr_images[step*batch_size:(step+1)*batch_size,:,:,:,:]
                hr = hr_images[step*batch_size:(step+1)*batch_size,:,:,:,:]
                logs = self.generator.train_on_batch(lr, hr)
                logs = cb.named_logs(self.generator, logs)
                print("  Step {}/{}: loss_G={:10.4f}".format(
                      step+1,step_per_epoch, logs['loss']), flush=True)

            for callback in callbacks:
                callback.on_epoch_end(epoch+1, logs)
