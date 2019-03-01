"""The MomentumIterativeMethod attack.
"""

import warnings

import numpy as np
import tensorflow as tf

# from cleverhans.attacks.attack import Attack
from cleverhans.attacks import Attack
from cleverhans.compat import reduce_sum, reduce_mean, softmax_cross_entropy_with_logits
from cleverhans import utils_tf
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  interval = (2*nsig+1.)/(kernlen)
  x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw/kernel_raw.sum()
  return kernel

class GradSmoothMomentumIterativeMethod(Attack):
  """
  The Momentum Iterative Method (Dong et al. 2017). This method won
  the first places in NIPS 2017 Non-targeted Adversarial Attacks and
  Targeted Adversarial Attacks. The original paper used hard labels
  for this attack; no label smoothing.
  Paper link: https://arxiv.org/pdf/1710.06081.pdf

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a MomentumIterativeMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(GradSmoothMomentumIterativeMethod, self).__init__(model, sess, dtypestr,
                                                  **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max')
    self.structural_kwargs = [
        'ord', 'nb_iter', 'decay_factor', 'sanity_checks']

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: Keyword arguments. See `parse_params` for documentation.
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)
  
    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(x,
                                                   tf.cast(self.clip_min,
                                                           x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x,
                                                tf.cast(self.clip_max,
                                                        x.dtype)))

    # Initialize loop variables
    momentum = tf.zeros_like(x)
    adv_x = x

    # Fix labels to the first model predictions for loss computation
    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def cond(i, _, __):
      return tf.less(i, self.nb_iter)

    def body(i, ax, m):
      logits = self.model.get_logits(ax)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      # print("body", loss, ax)

      # Define gradient of loss wrt input
      grad, = tf.gradients(loss, ax)

      grad = self.grad_smooth(grad)

      # Normalize current gradient and add it to the accumulated gradient
      grad = self.grad_norm(grad)

      #momentom
      m = self.decay_factor * m + grad

      m = self.grad_norm(m)

      optimal_perturbation = optimize_linear(m, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      # Update and clip adversarial example in current iteration
      ax = ax + optimal_perturbation
      ax = x + utils_tf.clip_eta(ax - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        ax = utils_tf.clip_by_value(ax, self.clip_min, self.clip_max)

      ax = tf.stop_gradient(ax)

      return i + 1, ax, m

    _, adv_x, _ = tf.while_loop(
        cond, body, (tf.zeros([]), adv_x, momentum), back_prop=True,
        maximum_iterations=self.nb_iter)

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x
  
  def grad_norm(self, grad):
    batch_size = tf.shape(grad)[0]
    avoid_zero_div = tf.cast(1e-12, grad.dtype)
    std = tf.reshape(tf.contrib.keras.backend.std(tf.reshape(grad, [batch_size, -1]), axis=1), [batch_size, 1, 1, 1])
    std = tf.maximum(avoid_zero_div, std)
    return grad/std

  def grad_smooth(self, grad):
    sig = 4
    kernel = gkern(7, sig).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    
    grad = tf.nn.depthwise_conv2d(grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    return grad

  def parse_params(self,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   decay_factor=1.0,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param decay_factor: (optional) Decay factor for the momentum term.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # Save attack-specific parameters
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.decay_factor = decay_factor
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True

def optimize_linear(grad, eps, ord=np.inf):
  if ord == np.inf:
    optimal_perturbation = tf.clip_by_value(tf.round(grad), -2, 2)
  else:
    raise NotImplementedError("Only L-inf, norms are "
                              "currently implemented.")

  scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)
  return scaled_perturbation