import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np

import io
        
def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
      result[l[0]] = l[1]
    return result

class GanLogs(Callback):
    def __init__(self, model, logpath, modeltype='generator'):
        self.logpath = logpath
        self.model = model
        self.modeltype = modeltype
    def on_epoch_end(self, epoch, logs={}):
        if self.modeltype=='generator':
            logger1 = tf.summary.create_file_writer(self.logpath + '/scalar/loss_G')
            logger2 = tf.summary.create_file_writer(self.logpath + '/scalar/adversarial')
            logger3 = tf.summary.create_file_writer(self.logpath + '/scalar/pixel_loss')
            logger4 = tf.summary.create_file_writer(self.logpath + '/scalar/energy_loss')
            logger5 = tf.summary.create_file_writer(self.logpath + '/scalar/flux_loss')
            logger6 = tf.summary.create_file_writer(self.logpath + '/scalar/enstrophy_loss')
            logger7 = tf.summary.create_file_writer(self.logpath + '/scalar/PSNR')
            with logger1.as_default():
                tf.summary.scalar(name='Generator Loss', data=logs['loss'], step=epoch)
            with logger2.as_default():
                tf.summary.scalar(name='Generator Loss', data=logs['gen'], step=epoch)
            with logger3.as_default():
                tf.summary.scalar(name='Generator Loss', data=logs['pixel'], step=epoch)
            with logger4.as_default():
                tf.summary.scalar(name='Generator Loss', data=logs['total_energy'], step=epoch)
            with logger5.as_default():
                tf.summary.scalar(name='Generator Loss', data=logs['mass_flux'], step=epoch)
            with logger6.as_default():
                tf.summary.scalar(name='Generator Loss', data=logs['enstrophy'], step=epoch)
            with logger7.as_default():
                tf.summary.scalar(name='PSNR', data=logs['PSNR'], step=epoch)

            return
        elif self.modeltype=='discriminator':
            logger1 = tf.summary.create_file_writer(self.logpath + '/scalar/loss_D')
            logger2 = tf.summary.create_file_writer(self.logpath + '/scalar/output_real')
            logger3 = tf.summary.create_file_writer(self.logpath + '/scalar/output_fake')
            with logger1.as_default():
                tf.summary.scalar(name='Discriminator Loss', data=logs['loss'], step=epoch)
            with logger2.as_default():
                tf.summary.scalar(name='Discriminator Output', data=logs['output_real'], step=epoch)
            with logger3.as_default():
                tf.summary.scalar(name='Discriminator Output', data=logs['output_fake'], step=epoch)
    
class GeneratorLogs(Callback):
    def __init__(self, model, logpath):
        self.logpath = logpath
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        logger1 = tf.summary.create_file_writer(self.logpath + '/scalar/loss')
        logger2 = tf.summary.create_file_writer(self.logpath + '/scalar/pixel_loss')
        logger3 = tf.summary.create_file_writer(self.logpath + '/scalar/energy_loss')
        logger4 = tf.summary.create_file_writer(self.logpath + '/scalar/flux_loss')
        logger5 = tf.summary.create_file_writer(self.logpath + '/scalar/enstrophy_loss')
        logger6 = tf.summary.create_file_writer(self.logpath + '/scalar/PSNR')
        with logger1.as_default():
            tf.summary.scalar(name='Generator Loss', data=logs['loss'], step=epoch)
        with logger2.as_default():
            tf.summary.scalar(name='Generator Loss', data=logs['pixel'], step=epoch)
        with logger3.as_default():
            tf.summary.scalar(name='Generator Loss', data=logs['total_energy'], step=epoch)
        with logger4.as_default():
            tf.summary.scalar(name='Generator Loss', data=logs['mass_flux'], step=epoch)
        with logger5.as_default():
            tf.summary.scalar(name='Generator Loss', data=logs['enstrophy'], step=epoch)
        with logger6.as_default():
            tf.summary.scalar(name='PSNR', data=logs['PSNR'], step=epoch)
        return
    
    
def imageGenerator(lr, sr, hr):
    nC = lr.shape[-1]
    figure, ax = plt.subplots(3, nC, figsize=(8,5))
    titles=[r'$u$',r'$v$',r'$w$',r'$\log \rho$',r'$\log P$']
    cmaps = [plt.cm.RdBu, plt.cm.RdBu, plt.cm.RdBu, plt.cm.viridis,plt.cm.viridis]
    ax[0,0].set_ylabel('Filtered')
    ax[1,0].set_ylabel('PISRT_GAN')
    ax[2,0].set_ylabel('Unfiltered')
    for i in range(nC):
        vmin = np.min(hr[:,:,i])
        vmax = np.max(hr[:,:,i])
        for j in range(3):
            ax[j,i].set_xticks([])
            ax[j,i].set_yticks([])
            ax[j,i].grid(False)
            if j==0:
                ax[j,i].set_title(titles[i])
                ax[j,i].imshow(lr[:,:,i], cmap=cmaps[i], vmin=vmin, vmax=vmax)
            if j==1:
                ax[j,i].imshow(sr[:,:,i], cmap=cmaps[i], vmin=vmin, vmax=vmax)
            if j==2:
                ax[j,i].imshow(hr[:,:,i], cmap=cmaps[i], vmin=vmin, vmax=vmax)
    return figure

class ImgCallback(Callback):
    def __init__(self, logpath, generator, lr_data, hr_data):
        self.generator = generator
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.logpath = logpath

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside a notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def on_epoch_end(self, epoch, logs={}):
        lr = self.lr_data
        hr = self.hr_data

        sr = self.generator.predict(lr, steps=1)

        lr = lr[0,:,:,0,:]
        sr = sr[0,:,:,0,:]
        hr = hr[0,:,:,0,:]
        figure = imageGenerator(lr, sr, hr)
        file_writer = tf.summary.create_file_writer(self.logpath)

        with file_writer.as_default():
            tf.summary.image('Channel Slices', self.plot_to_image(figure), step=epoch)

        return
