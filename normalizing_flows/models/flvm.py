import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flows as flows
import models.utils as utils
from tqdm import tqdm
from .trackable_module import TrackableModule

class FlowLVM(TrackableModule):
    """
    Flow-based Latent Variable Model; attempts to learn a variational approximation
    for the joint distribution p(x,z) by minimizing the log likelihood of F^-1(x)
    under the prior p(z) where F is an invertible transformation z <--> x.
    """
    def __init__(self,
                 transform: flows.transform.Transform,
                 prior: tfp.distributions.Distribution=tfp.distributions.Normal(loc=0.0, scale=1.0),
                 dim=2,
                 input_channels=None,
                 num_bins=16,
                 cond_fn=None,
                 optimizer_flow=tf.keras.optimizers.Adam(lr=1.0E-4),
                 optimizer_cond=None,
                 clip_grads=1.0,
                 name='flvm'):
        """
        transform     : a bijective transform to be applied to the initial variational density;
                        note that this is assumed to be a transform z -> x where the inverse is x -> z
        prior         : a tfp.distributions.Distribution representing the prior, p(z)
        input_shape   : the shape of the observed variables, x
        num_bins      : for discrete input spaces: number of discretized bins; i.e. num_bins = 2^(num_bits)
        optimizer     : optimizer to use during training
        clip_grads    : If not None and > 0, the gradient clipping ratio for clip_by_global_norm;
                        otherwise, no gradient clipping is applied
        """
        super().__init__({'optimizer_flow': optimizer_flow}, name=name)
        if optimizer_cond is not None:
            super().__init__({'optimizer_cond': optimizer_cond}, name=name)
        self.prior = prior
        self.transform = transform
        self.dim = dim
        self.num_bins = num_bins

        self.optimizer_flow = optimizer_flow
        self.optimizer_cond = optimizer_cond
        self.clip_grads = clip_grads
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 0.0
        self.input_channels = input_channels
        if self.input_channels is not None:
            input_shape = tf.TensorShape((None,*[None for i in range(self.dim)], self.input_channels))
            self.initialize(input_shape)


    def initialize(self, input_shape):
        self.input_shape = input_shape
        with tf.init_scope():
            self.transform.initialize(input_shape)
            self.cond_fn = self.transform._cond_fn

    @tf.function
    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, -.5/self.num_bins, .5/self.num_bins, dtype=tf.float32)
        #x += tf.random.normal(x.shape, 0, 0.01, dtype=tf.float32)
        return x

    @tf.function
    def eval_cond(self, x, y, **kwargs):
        z = self.prior.sample(x.shape)
        pred, _ = self.transform.forward(z, y_cond=y, **kwargs)
        return tf.math.reduce_mean(tf.math.abs(x - pred))

    @tf.function
    def eval_batch(self, x, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        num_elements = tf.cast(tf.math.reduce_prod(x.shape[1:]), dtype=tf.float32)
        x = self._preprocess(x)
        z, ldj = self.transform.inverse(x, **flow_kwargs)
        y_loss = 0.0
        #if self.cond_fn is not None and 'y_cond' in flow_kwargs:
        #    y_loss = self.eval_cond(x, flow_kwargs['y_cond'])

        prior_log_probs = self.prior.log_prob(z)
        if prior_log_probs.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_log_probs = tf.math.reduce_sum(prior_log_probs, axis=[i for i in range(1,z.shape.rank)])
        log_probs = prior_log_probs + ldj

        nll = -(log_probs - self.scale_factor*num_elements) / num_elements
        return tf.math.reduce_mean(nll), \
               tf.math.reduce_mean(-ldj / num_elements), \
               tf.math.reduce_mean(prior_log_probs / num_elements), \
               y_loss

    @tf.function
    def train_batch(self, x, **flow_kwargs):
        """
        Performs a single iteration of mini-batch SGD on input x.
        Returns loss, nll, prior, nldj[, grad_norm]
                where loss is the total optimized loss (including regularization),
                nll is the averaged negative log likelihood component,
                prior is the averaged prior negative log likelihodd,
                nldj is the negative log det jacobian,
                and, if clip_grads is True, grad_norm is the global max gradient norm
        """
        assert self.input_shape is not None, 'model not initialized'
        nll, nldj, prior_log_probs, y_loss = self.eval_batch(x, **flow_kwargs)
        reg_loss = self.transform._regularization_loss()
        loss_flow = nll #+ reg_loss
        loss_cond = y_loss
        train_var = self.trainable_variables

        gradients = tf.gradients(loss_flow, train_var)
        if self.clip_grads:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grads)
        self.optimizer_flow.apply_gradients(zip(gradients, train_var))

        return loss_flow, y_loss, nll, nldj, prior_log_probs

    def train(self, train_data: tf.data.Dataset, steps_per_epoch, num_epochs=1, conditional=False, init=False, **flow_kwargs):
        train_data = train_data.take(steps_per_epoch).repeat(num_epochs)
        hist = dict()
        init = tf.constant(init) # init variable for data-dependent initialization
        with tqdm(total=steps_per_epoch*num_epochs) as prog:
            for epoch in range(num_epochs):
                for i, batch in enumerate(train_data.take(steps_per_epoch)):
                    if conditional:
                        x, y = batch
                        loss_flow, loss_cond, nll, nldj, p  = self.train_batch(x, y_cond=y, init=init, **flow_kwargs)
                        utils.update_metrics(hist, lf=loss_flow.numpy(), p=p.numpy(), nldj=nldj.numpy())
                    else:
                        x = batch
                        loss_flow, _, nll, nldj, p  = self.train_batch(x, init=init, **flow_kwargs)
                        utils.update_metrics(hist, lf=loss_flow.numpy(), p=p.numpy(), nldj=nldj.numpy())
                    init=tf.constant(False)
                    prog.set_postfix({k: v[0] for k,v in hist.items()})
                    prog.update(1)

    def evaluate(self, validation_data: tf.data.Dataset, validation_steps, conditional=False, **flow_kwargs):
        validation_data = validation_data.take(validation_steps)
        with tqdm(total=validation_steps) as prog:
            hist = dict()
            for batch in validation_data:
                if conditional:
                    x, y = batch
                    nll, ldj, y_loss  = self.eval_batch(x, y_cond=y, **flow_kwargs)
                else:
                    x = batch
                    nll, ldj, y_loss  = self.eval_batch(x, **flow_kwargs)
                utils.update_metrics(hist, nll=nll.numpy())
                prog.update(1)
                prog.set_postfix({k: v[0] for k,v in hist.items()})

    def encode(self, x, y_cond=None):
        if y_cond is not None:
            z, _ = self.transform.inverse(x, y_cond=y_cond)
        else:
            z, _ = self.transform.inverse(x)
        return z

    def decode(self, z, y_cond=None):
        if y_cond is not None:
            x, _ = self.transform.forward(z, y_cond=y_cond)
        else:
            x, _ = self.transform.forward(z)
        return x

    def sample(self, n=1, shape=32, tau=1.0, y_cond=None):
        assert self.input_shape is not None, 'model not initialized'
        batch_size = 1 if y_cond is None else y_cond.shape[0]
        event_ndims = self.prior.event_shape.rank
        shape = (1,*[shape for _ in range(self.dim)], self.input_channels)
        z_shape = shape[1:]

        z = tau*self.prior.sample((n*batch_size,*z_shape[:len(z_shape)-event_ndims]))
        z = tf.reshape(z, (n*batch_size, -1))
        if y_cond is not None:
            # repeat y_cond n times along batch axis
            y_cond = tf.repeat(y_cond, n, axis=0)
        ret = self.decode(z, y_cond=y_cond).numpy()
        return ret

    def param_count(self):
        return self.transform.param_count()

    def test(self, shape, **kwargs):
        if 'y_shape' in kwargs and kwargs['y_shape'] is not None:
            y_shape = kwargs['y_shape'].copy()
            y_shape = (1,*[y_shape for _ in range(self.dim)], self.cond_channels)

        shape = (1,*[shape for _ in range(self.dim)], self.input_channels)

        self.transform._test(shape, **kwargs)

        print('Testing full model:')
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        x = normal.sample(shape)
        if 'y_shape' in kwargs and self.transform._cond_fn is not None:
            normal = tfp.distributions.Normal(loc=0, scale=1.0)
            y_cond = normal.sample(y_shape)
        else:
            y_cond = None

        z, fldj = self.transform.forward(x, y_cond=y_cond)
        assert np.all(np.isfinite(z)), 'forward has nan output'
        x_, ildj = self.transform.inverse(z, y_cond=y_cond)
        err_x = tf.reduce_mean(x_-x)
        err_ldj = tf.reduce_mean(ildj+fldj)
        print("\tError on forward inverse pass:")
        print("\t\tx-F^{-1}oF(x):", err_x.numpy())
        print("\t\tildj+fldj:", err_ldj.numpy())
        print('\tpassed')
