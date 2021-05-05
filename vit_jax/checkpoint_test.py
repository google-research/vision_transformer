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

import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jnp

from vit_jax import checkpoint
from vit_jax import models
from vit_jax import test_utils
from vit_jax.configs import models as config_lib


class CheckpointTest(absltest.TestCase):

  def test_load_pretrained(self):
    tempdir = tempfile.gettempdir()
    model_config = config_lib.get_testing_config()
    test_utils.create_checkpoint(model_config, f'{tempdir}/testing.npz')
    model = models.VisionTransformer(num_classes=2, **model_config)
    variables = model.init(
        jax.random.PRNGKey(0),
        inputs=jnp.ones([1, 32, 32, 3], jnp.float32),
        train=False,
    )
    checkpoint.load_pretrained(
        pretrained_path=f'{tempdir}/testing.npz',
        init_params=variables['params'],
        model_config=model_config)

  def test_convert_pre_linen_pytree(self):
    params = checkpoint.convert_pre_linen_pytree({
        'mod_0': {
            'submod1_0': {},
            'submod2_1': {},
            'submod1_2': {},
        },
        'mod2_2': {
            'mod2_2_0': {}
        },
        'mod2_11': {
            'mod2_11_0': {}
        },
        'mod2_1': {
            'mod2_1_0': {}
        },
    })
    self.assertDictEqual(params, {
        'mod_0': {
            'submod1_0': {},
            'submod1_1': {},
            'submod2_0': {},
        },
        'mod2_0': {
            'mod2_1_0': {}
        },
        'mod2_1': {
            'mod2_2_0': {}
        },
        'mod2_2': {
            'mod2_11_0': {}
        },
    })


if __name__ == '__main__':
  absltest.main()
