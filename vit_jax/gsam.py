import jax
import jax.numpy as jnp
from vit_jax import utils

def dual_vector(y, ):
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
  Args:
      y: A pytree of numpy ndarray, vector y in the equation above.
      sign: -1.0 or 1.0
  """
  gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient, gradient_norm

def gsam_gradient(loss_fn, base_opt, inputs, targets, grad_accum_steps,
                  rho_max, rho_min, alpha, lr, lr_max, lr_min=0.0, eps=1e-12):
  """
  Get the GSAM gradient (https://openreview.net/pdf?id=edONMAnhLu-) of the loss function.
  Args: 
    loss_fn: the loss function.
    base_opt: the base optimizer.
    inputs: the inputs to the loss function.
    targets: the targets to the loss function.
    grad_accum_steps: the number of steps to accumulate the gradient.
    rho_max: the maximum rho value for perturbation of weights.
    rho_min: the minimum rho value for perturbation of weights.
    alpha: the alpha value for the rho schedule, see Algorithm 1 in the paper.
    lr: current learning rate.
    lr_max: the maximum learning rate.
    lr_min: the minimum learning rate.
    eps: the epsilon value for numerical stability.

  Returns:
    l_clean: the loss function value.
    g_gsam: the GSAM gradient. g_gsam is not averaged across workers, need to call "jax.lax.pmean" to average.
      
  Note:
    Setting `rho_max=rho_min` and `alpha=0` reduces GSAM to SAM.
  """
  l_clean, g_clean = utils.accumulate_gradient(jax.value_and_grad(loss_fn), base_opt.target,
                                inputs, targets, grad_accum_steps)
  g_clean_normalized, g_clean_length = dual_vector(g_clean)

  if lr_max == lr_min:
    sam_rho = rho_max
  else:
    sam_rho = rho_min + (rho_max - rho_min) * (lr - lr_min) / (lr_max - lr_min)
    
  # per-worker perturbation.
  param_sam = jax.tree_multimap(lambda a, b: a + sam_rho * b / (g_clean_length + eps),
                                base_opt.target, g_clean)

  # get gradients at perturbed weights.
  l_robust, g_robust = utils.accumulate_gradient(jax.value_and_grad(loss_fn), param_sam,
                                inputs, targets, grad_accum_steps)

  # decompose gradients.
  g_clean_flatten, _ = jax.tree_flatten(g_clean)

  g_robust_normalized, g_robust_length = dual_vector(g_robust)
  g_robust_normalized_flatten, _ = jax.tree_flatten(g_robust_normalized)

  g_clean_projection_norm = sum([jnp.vdot(p, q) for (p,q) in
      zip(g_robust_normalized_flatten, g_clean_flatten)])
  g_clean_residual = jax.tree_multimap(lambda a, b:
      a - g_clean_projection_norm * b, g_clean,g_robust_normalized)

  # get GSAM gradient.
  g_gsam = jax.tree_multimap( lambda a, b: a - b * alpha, 
      g_robust, g_clean_residual)
      
  return l_clean, g_gsam
