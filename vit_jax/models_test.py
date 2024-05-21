# Copyright 2024 Google LLC.
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
    'LiT-B16B': 195_871_489,
    'LiT-B16B_2': 195_280_897,
    'LiT-L16L': 638_443_521,
    'LiT-L16S': 331_140_353,
    'LiT-L16Ti': 311_913_089.,
    'Mixer-B_16': 59_880_472,
    'Mixer-B_32': 60_293_428,
    'Mixer-L_16': 208_196_168,
    'R+ViT-Ti_16': 6_337_704,
    'R26+ViT-B_32': 101_383_976,
    'R26+ViT-S_32': 36_431_912,
    'R50+ViT-B_16': 98_659_112,
    'R50+ViT-L_32': 328_994_856,
    'ViT-B_8': 86_576_872,
    'ViT-B_16': 86_567_656,
    'ViT-B_16-gap-norep': 86_566_120,
    'ViT-B_32': 88_224_232,
    'ViT-B_32-gap-norep': 88_222_696,
    'ViT-H_14': 632_045_800,
    'ViT-L_16': 304_326_632,
    'ViT-L_32': 306_535_400,
    'ViT-S_16': 22_050_664,
    'ViT-S_16-gap-norep': 22_049_896,
    'ViT-S_32': 22_878_952,
    'ViT-S_32-gap-norep': 22_878_184,
    'ViT-Ti_16': 5_717_416,
    'testing': 21_390,
    'testing-unpooled': 21_370,
}


class ModelsTest(parameterized.TestCase):

  def test_all_tested(self):
    self.assertEmpty(set(config_lib.MODEL_CONFIGS).difference(MODEL_SIZES))

  @parameterized.parameters(*list(MODEL_SIZES.items()))
  def test_can_instantiate(self, name, size):
    rng = jax.random.PRNGKey(0)
    kw = {} if name.startswith('LiT-') else dict(num_classes=1_000)
    model = models.get_model(name, **kw)
    batch_size = 2
    images = jnp.ones([batch_size, 224, 224, 3], jnp.float32)
    if name.startswith('LiT-'):
      tokens = jnp.ones([batch_size, model.pp.max_len], jnp.int32)
      variables = model.init(rng, images=images, tokens=tokens)
      zimg, ztxt, _ = model.apply(variables, images=images, tokens=tokens)
      self.assertEqual(zimg.shape[0], batch_size)
      self.assertEqual(zimg.shape, ztxt.shape)
    else:
      variables = model.init(rng, images, train=False)
      outputs = model.apply(variables, images, train=False)
      if 'unpooled' in name:
        self.assertEqual((2, 196, 1000), outputs.shape)
      else:
        self.assertEqual((2, 1000), outputs.shape)
    param_count = sum(p.size for p in jax.tree.flatten(variables)[0])
    self.assertEqual(
        size, param_count,
        f'Expected {name} to have {size} params, found {param_count}.')


if __name__ == '__main__':
  absltest.main()
