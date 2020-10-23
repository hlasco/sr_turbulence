import tensorflow as tf
import tensorflow_probability as tfp
from typing import List
from .transform import Transform, AmortizedTransform

class Flow(AmortizedTransform):
    def __init__(self, steps: List[Transform], input_shape=None, name='flow', *args, **kwargs):
        """
        Constructs a new flow as a sequence of transforms or sub-flows.
        """
        # unroll nested flows
        steps_acc = []
        for step_i in steps:
            if isinstance(step_i, Flow):
                for step_j in step_i.steps:
                    steps_acc.append(step_j)
            else:
                steps_acc.append(step_i)
        self.steps = steps_acc
        self.num_steps = len(self.steps)
        # add num_flows alias for legacy code
        self.num_flows = self.num_steps
        super().__init__(*args, input_shape=input_shape, name=name, **kwargs)

    @staticmethod
    def uniform(num_flows, transform_init):
        """
        Creates a simple, uniform flow with 'num_flows' steps using the transform_init constructor function.
        transform_init should follow the signature f: i -> Transform, where i is the index of the current step
        in the flow sequence and Transform is a valid transformer instance.
        """
        assert num_flows > 0, "num_flows must be > 0"
        transforms = [transform_init(i) for i in range(num_flows)]
        transform_type = type(transforms[0])
        assert all([transform_type == type(t) for t in transforms]), "All transforms should have the same type for uniform flow"
        return Flow(transforms)

    def _forward_shape(self, input_shape):
        for step in self.steps:
            input_shape = step._forward_shape(input_shape)
        return input_shape

    def _inverse_shape(self, input_shape):
        for step in reversed(self.steps):
            input_shape = step._inverse_shape(input_shape)
        return input_shape

    def _initialize(self, input_shape):
        for step in self.steps:
            step.initialize(input_shape)
            input_shape = step._forward_shape(input_shape)

    def _forward(self, z_0, *params: tf.Tensor, return_sequence=False, **kwargs):
        """
        Computes the forward pass of the flow: z_k = f_k . f_k-1 ... f_1(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        params : optional sequence of tensors (batch_size, m_i) where m_i is the number of parameters for flow step i
        """
        zs = [z_0]
        ldj = 0.0
        for i in range(self.num_steps):
            step = self.steps[i]
            params_i = [params[i]] if len(params) > i else []
            z_i, ldj_i = step.forward(zs[-1], *params_i, **kwargs)
            zs.append(z_i)
            ldj += ldj_i
        return (zs, ldj) if return_sequence else (zs[-1], ldj)

    def _inverse(self, z, *params: tf.Tensor, return_sequence=False, **kwargs):
        """
        Computes the inverse pass of the flow: z_0 = f^-1_1 . f^-1_2 ... f^-1_k(z)

        Tensor shapes:
        z_0    : (batch_size, d)
        params : optional sequence of tensors (batch_size, m_i) where m_i is the number of parameters for flow step i
        """
        zs = [z]
        ldj = 0.0
        for i in range(self.num_steps):
            step = self.steps[self.num_steps-i-1]
            params_i = [params[i]] if len(params) > i else []
            z_i, ldj_i = step.inverse(zs[-1], *params_i, **kwargs)
            tf.debugging.assert_all_finite(z_i, f'{step.name} output nan/inf values for input {zs[-1]}')
            zs.append(z_i)
            ldj += ldj_i
        return (zs, ldj) if return_sequence else (zs[-1], ldj)

    def _regularization_loss(self):
        return tf.math.add_n([t.regularization_loss() for t in self.steps])

    def _param_count(self, shape):
        return tf.math.reduce_sum([t.param_count(shape) for t in self.steps])# if isinstance(t, AmortizedTransform)])

    def _create_variables(self, shape, initializer=None, **kwargs):
        return sum([t.create_variables(shape, initializer, **kwargs) \
                    for t in self.steps if isinstance(t, AmortizedTransform)],[])

    def _test(self, shape, **kwargs):
        for i in range(self.num_steps):
            step = self.steps[i]
            #params_i = [params[i]] if len(params) > i else []
            step._test(shape, **kwargs)
            print('\t Num params:', step.param_count(shape))
            shape = step._forward_shape(shape)
