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
                 prior = tfp.distributions.Normal(loc=0, scale=1.0),
                 dim=2,
                 input_channels=None,
                 cond_channels=None,
                 num_bins=None,
                 cond_fn=None,
                 learning_rate=1e-4,
                 clip_grads=.5,
                 rundir='',
                 name='flvm'):
        """
        transform     : a bijective transform to be applied to the initial variational density;
                        note that this is assumed to be a transform z -> x where the inverse is x -> z
        prior         : a tfp.distributions.Distribution representing the prior, p(z)
        dim           : the spatial dimension of input
        input_channels: the number of channels of the observed variables, x
        num_bins      : for discrete input spaces: number of discretized bins; i.e. num_bins = 2^(num_bits)
        cond_fn       : the conditioning function, for conditional mode
        optimizer_flow: optimizer to use during training the full model
        clip_grads    : If not None and > 0, the gradient clipping ratio for clip_by_global_norm;
                        otherwise, no gradient clipping is applied
        """
        #strategy=tf.distribute.MirroredStrategy()
        optimizer_flow = tf.keras.optimizers.Adam(lr=learning_rate)
        super().__init__({'optimizer_flow': optimizer_flow}, name=name)

        self.rundir=rundir
        self.prior = prior
        self.transform = transform
        self.dim = dim
        self.num_bins = num_bins

        self.optimizer_flow = optimizer_flow
        self.clip_grads = clip_grads
        self.scale_factor = np.log2(num_bins) if num_bins is not None else 0.0
        self.input_channels = input_channels
        self.cond_channels  = cond_channels
        if self.input_channels is not None:
            input_shape = tf.TensorShape((None,*[None for i in range(self.dim)], self.input_channels))
            self.initialize(input_shape)


    def initialize(self, input_shape):
        self.input_shape = input_shape
        with tf.init_scope():
            self.transform.initialize(input_shape)
            self.cond_fn = self.transform._cond_fn

    def _preprocess(self, x):
        if self.num_bins is not None:
            x += tf.random.uniform(x.shape, -.5/self.num_bins, .5/self.num_bins, dtype=tf.float32)
        return x

    def eval_cond(self, x_true, y, **kwargs):
        z = self.prior.sample(x_true.shape)*0
        x_pred, _ = self.transform.forward(z, y_cond=y, **kwargs)
        diff = x_true-x_pred
        loss = tf.math.reduce_mean(tf.abs(diff), axis=[i for i in range(1,diff.shape.rank)])

        return loss

    def eval_batch(self, x, batch_size, **flow_kwargs):
        assert self.input_shape is not None, 'model not initialized'
        num_elements = tf.cast(tf.math.reduce_prod(x.shape[1:]), dtype=tf.float32)
        x = self._preprocess(x)
        z, ldj = self.transform.inverse(x, **flow_kwargs)
        y_loss = tf.zeros(batch_size)
        #if self.cond_fn is not None and 'y_cond' in flow_kwargs:
        #    y_loss = self.eval_cond(x, flow_kwargs['y_cond'])

        #prior_log_probs = self.prior.log_prob(z, scale=p_scale)
        prior_log_probs = self.prior.log_prob(z)
        if prior_log_probs.shape.rank > 1:
            # reduce log probs along non-batch dimensions
            prior_log_probs = tf.math.reduce_sum(prior_log_probs, axis=[i for i in range(1,z.shape.rank)])
        log_probs = prior_log_probs + ldj

        nll = -(log_probs - self.scale_factor*num_elements) / num_elements
        return tf.nn.compute_average_loss(nll, global_batch_size=batch_size), \
               tf.nn.compute_average_loss(-ldj / num_elements, global_batch_size=batch_size), \
               tf.nn.compute_average_loss(prior_log_probs / num_elements, global_batch_size=batch_size), \
               tf.nn.compute_average_loss(y_loss, global_batch_size=batch_size)

    @tf.function
    def train_batch(self, x, batch_size, **flow_kwargs):
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
        nll, nldj, prior_log_probs, y_loss = self.eval_batch(x, batch_size, **flow_kwargs)
        #reg_loss = self.transform._regularization_loss()
        loss_flow = nll #+ y_loss #+ reg_loss
        train_var = self.trainable_variables

        gradients = tf.gradients(loss_flow, train_var)
        if self.clip_grads:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_grads)

        self.optimizer_flow.apply_gradients(zip(gradients, train_var))
        ret = tf.stack([loss_flow, y_loss, nll, nldj, prior_log_probs])
        return ret

    def train(self, train_data: tf.data.Dataset, batch_size, steps_per_epoch,
              num_epochs=1, epoch_0=0, conditional=False, init=False, strategy=tf.distribute.get_strategy(),
              **flow_kwargs):
        steps_per_epoch = steps_per_epoch
        global_batch_size = batch_size*strategy.num_replicas_in_sync
        init = tf.constant(init) # init variable for data-dependent initialization
        iterator = iter(train_data)
        hist = dict()
        for epoch in range(num_epochs):
            with tqdm(total=steps_per_epoch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as prog:
                for i in range(steps_per_epoch):

                    batch = iterator.next()
                    x = batch['x']
                    if conditional:
                        flow_kwargs['y_cond'] = batch['y']
                    else:
                        flow_kwargs['y_cond'] = tf.repeat(None, global_batch_size, axis=0)
                    flow_kwargs['init']=init
                    per_replica_metrics = strategy.run(self.train_batch, args=(x, global_batch_size,), kwargs=flow_kwargs)
                    per_replica_metrics = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metrics, axis=None)
                    loss_flow, y_loss, nll, nldj, p = per_replica_metrics

                    init=tf.constant(False)
                    utils.update_metrics(hist, lf=loss_flow.numpy(), ly=y_loss.numpy(), p=p.numpy(), nldj=nldj.numpy())

                    if i%100==0:
                        prog.set_postfix({k: v[0] for k,v in hist.items()})
                        prog.update(100)
                        num_step = (epoch_0+epoch)*steps_per_epoch + i
                        utils.update_metrics(hist, lf=loss_flow.numpy(), ly=y_loss.numpy(), p=p.numpy(), nldj=nldj.numpy())
                        utils.write_logs(hist, num_step, self.rundir)
                        # Reset metrics
                        hist = dict()
            self.save(self.rundir + 'model')

    def encode(self, x, y_cond=None, **kwargs):
        if y_cond is not None:
            z, _ = self.transform.inverse(x, y_cond=y_cond, **kwargs)
        else:
            z, _ = self.transform.inverse(x,**kwargs)
        return z

    def decode(self, z, y_cond=None, **kwargs):
        if y_cond is not None:
            x, _ = self.transform.forward(z, y_cond=y_cond, **kwargs)
        else:
            x, _ = self.transform.forward(z, **kwargs)
        return x

    def sample(self, n=1, shape=32, y_cond=None, prior_loc=0.0, prior_scale=1.0):
        assert self.input_shape is not None, 'model not initialized'
        batch_size = 1 if y_cond is None else y_cond.shape[0]
        event_ndims = self.prior.event_shape.rank
        shape = (1,*[shape for _ in range(self.dim)], self.input_channels)
        z_shape = shape[1:]
        
        z = self.prior.sample((n*batch_size,*z_shape))
        #z = tf.reshape(z, (n*batch_size, -1))
        if y_cond is not None:
            # repeat y_cond n times along batch axis
            y_cond = tf.repeat(y_cond, n, axis=0)
        ret = self.decode(prior_scale*z, y_cond=y_cond).numpy()
        return ret

    def param_count(self):
        return self.transform.param_count()

    def test(self, shape, **kwargs):
        if 'y_shape' in kwargs and kwargs['y_shape'] is not None:
            y_shape = kwargs['y_shape']#.copy()
            y_shape = (1,*[y_shape for _ in range(self.dim)], self.cond_channels)

        shape = (1,*[shape for _ in range(self.dim)], self.input_channels)

        self.transform._test(shape, **kwargs)

        print('Testing full model:')
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        z = normal.sample(shape)
        if 'y_shape' in kwargs and self.transform._cond_fn is not None:
            normal = tfp.distributions.Normal(loc=0.0, scale=0.0)
            y_cond = normal.sample(y_shape)
        else:
            y_cond = None

        x, ildj = self.transform.inverse(z, y_cond=y_cond)
        assert np.all(np.isfinite(x)), 'inverse has nan output'
        z_, fldj = self.transform.forward(x, y_cond=y_cond)
        assert np.all(np.isfinite(z_)), 'forward has nan output'
        err_x = tf.reduce_mean(z_-z)
        err_ldj = tf.reduce_mean(ildj+fldj)
        print("\tError on forward inverse pass:")
        print("\t\tx-F^{-1}oF(x):", err_x.numpy())
        print("\t\tildj+fldj:", err_ldj.numpy())
        print('\tpassed')
