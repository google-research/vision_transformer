# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from vit_jax import models
from vit_jax.configs import models as config_lib


MODEL_SIZES = {
    'ViT-B_16': 86_567_656,
    'R50+ViT-B_16': 98_659_112,
    'ViT-B_32': 88_224_232,
    'R26+ViT-B_32': 101_383_976,
    'ViT-L_16': 304_326_632,
    'ViT-L_32': 306_535_400,
    'R50+ViT-L_32': 328_994_856,
    'ViT-H_14': 632_045_800,
}


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(*list(MODEL_SIZES.items()))
  def test_can_instantiate(self, name, size):
    rng = jax.random.PRNGKey(0)
    config = config_lib.MODEL_CONFIGS[name]
    model_cls = models.VisionTransformer
    model = model_cls(num_classes=1_000, **config)
    inputs = jnp.ones([2, 224, 224, 3], jnp.float32)
    variables = model.init(rng, inputs, train=False)
    outputs = model.apply(variables, inputs, train=False)
    self.assertEqual((2, 1000), outputs.shape)
    param_count = sum(p.size for p in jax.tree_flatten(variables)[0])
    self.assertEqual(
        size, param_count,
        f'Expected {name} to have {size} params, found {param_count}.')


if __name__ == '__main__':
  absltest.main()
