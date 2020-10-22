# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flax
import jax
import jax.numpy as jnp
import numpy as np


class Optimizer(flax.optim.OptimizerDef):
  """Momentum optimizer that stores state using half-precision."""

  @flax.struct.dataclass
  class HyperParams:
    learning_rate: np.ndarray
    beta: np.ndarray
    grad_norm_clip: np.ndarray

  @flax.struct.dataclass
  class State:
    momentum: np.ndarray

  def __init__(self, learning_rate=None, beta=0.9, grad_norm_clip=None):
    hyper_params = Optimizer.HyperParams(learning_rate, beta, grad_norm_clip)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return Optimizer.State(jnp.zeros_like(param, dtype=jnp.bfloat16))

  def apply_gradient(self, hyper_params, params, state, grads):
    step = state.step
    params_flat, treedef = jax.tree_flatten(params)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)

    # Optionally resize the global gradient to a maximum norm.
    if hyper_params.grad_norm_clip:
      grads_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
      grads_factor = jnp.minimum(1.0, hyper_params.grad_norm_clip / grads_l2)
      grads_flat = jax.tree_map(lambda param: grads_factor * param, grads_flat)

    out = [
        self.apply_param_gradient(step, hyper_params, param, state, grad)
        for param, state, grad in zip(params_flat, states_flat, grads_flat)
    ]

    new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
    new_params = jax.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
    new_state = flax.optim.OptimizerState(step + 1, new_param_states)
    return new_params, new_state

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + grad
    new_param = param - hyper_params.learning_rate * new_momentum
    new_state = Optimizer.State(new_momentum.astype(jnp.bfloat16))
    return new_param, new_state
