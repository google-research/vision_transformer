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

import unittest

import flax
import jax
import jax.numpy as jnp

from vit_jax import models

MODEL_SIZES = {
    'ViT-B_16': 86_567_656,
    'ViT-B_32': 88_224_232,
    'ViT-L_16': 304_326_632,
    'ViT-L_32': 306_535_400,
    'testing': 2985
}


class ModelsTest(unittest.TestCase):

  def test_can_instantiate(self):
    rng = jax.random.PRNGKey(0)
    for name, model in models.KNOWN_MODELS.items():
      output, initial_params = model.partial(num_classes=1000).init_by_shape(
          rng, [((2, 224, 224, 3), jnp.float32)])
      self.assertEqual((2, 1000), output.shape)
      param_count = sum(p.size for p in jax.tree_flatten(initial_params)[0])
      self.assertEqual(MODEL_SIZES[name], param_count)


if __name__ == '__main__':
  unittest.main()
