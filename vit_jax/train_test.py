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

import tempfile
import unittest

import flax
import jax
import jax.numpy as jnp
import os

from vit_jax import checkpoint_test
from vit_jax import flags
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import train


class TrainTest(unittest.TestCase):

  def test_main(self):
    basedir = tempfile.gettempdir()
    logdir = f'{basedir}/logs'
    copytodir = f'{basedir}/copyto'
    output = f'{basedir}/output/model.npz'
    checkpoint_test.create_checkpoint('testing', f'{basedir}/testing.npz')
    name = 'TrainTest.test_main'
    argv = [
        f'--name={name}',
        f'--model=testing',
        f'--vit_pretrained_dir={basedir}',
        f'--logdir={logdir}',
        f'--output={output}',
        f'--copy_to={copytodir}',
        '--dataset=cifar10',
        '--batch=8',
        '--accum_steps=2',
        '--batch_eval=8',
        '--total_steps=1',
    ]
    parser = flags.argparser(models.KNOWN_MODELS.keys(),
                             input_pipeline.DATASET_PRESETS.keys())
    args = parser.parse_args(argv)
    train.main(args)
    self.assertTrue(os.path.exists(f'{logdir}/{name}/train.log'))
    self.assertTrue(os.path.exists(f'{copytodir}/{name}/train.log'))
    self.assertTrue(os.path.exists(output))
    self.assertTrue(
        os.path.exists(f'{copytodir}/{name}/{os.path.basename(output)}'))


if __name__ == '__main__':
  unittest.main()
