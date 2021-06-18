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

import os
import tempfile
from absl.testing import absltest

import chex
import jax
import ml_collections
import tensorflow_datasets as tfds

from vit_jax import test_utils
from vit_jax import train
from vit_jax.configs import common
from vit_jax.configs import models


class TrainTest(absltest.TestCase):

  def test_train_and_evaluate(self):
    config = common.get_config()
    config.model = models.get_testing_config()
    config.dataset = 'cifar10'
    config.pp = ml_collections.ConfigDict(
        {'train': 'train[:98%]', 'test': 'test', 'resize': 448, 'crop': 384})
    config.batch = 64
    config.accum_steps = 2
    config.batch_eval = 8
    config.total_steps = 1

    with tempfile.TemporaryDirectory() as workdir:

      config.pretrained_dir = workdir
      test_utils.create_checkpoint(config.model, f'{workdir}/testing.npz')

      opt_pmap = train.train_and_evaluate(config, workdir)
      self.assertTrue(os.path.exists(f'{workdir}/checkpoint_1'))


if __name__ == '__main__':
  absltest.main()
