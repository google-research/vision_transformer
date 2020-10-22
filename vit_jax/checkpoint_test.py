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

import logging
import unittest

import jax
import jax.numpy as jnp

from vit_jax import checkpoint
from vit_jax import models


def create_checkpoint(model, path):
  model = models.KNOWN_MODELS[model].partial(num_classes=1)
  _, params = model.init_by_shape(
      jax.random.PRNGKey(0),
      [((1, 16, 16, 3), jnp.float32)],
  )
  checkpoint.save(params, path)


class CheckpointTest(unittest.TestCase):

  def test_load_pretrained(self):
    create_checkpoint('testing', './testing.npz')
    model = models.KNOWN_MODELS['testing'].partial(num_classes=2)
    _, params = model.init_by_shape(
        jax.random.PRNGKey(0),
        [((1, 32, 32, 3), jnp.float32)],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    checkpoint.load_pretrained(
        pretrained_path='testing.npz',
        init_params=params,
        model_config=models.CONFIGS['testing'],
        logger=logger)


if __name__ == '__main__':
  unittest.main()
