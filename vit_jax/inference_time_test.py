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

import glob
import tempfile

from absl.testing import absltest

from vit_jax import inference_time
from vit_jax import test_utils
from vit_jax.configs import inference_time as config_lib
from vit_jax.configs import models


class InferenceTimeTest(absltest.TestCase):

  def test_main(self):
    config = config_lib.get_config()
    config.num_classes = 10
    config.image_size = 224
    config.batch = 8
    config.model_name = 'testing'
    model_config = models.get_testing_config()

    workdir = tempfile.gettempdir()
    config.pretrained_dir = workdir
    test_utils.create_checkpoint(model_config, f'{workdir}/testing.npz')
    inference_time.inference_time(config, workdir)
    self.assertNotEmpty(glob.glob(f'{workdir}/events.out.tfevents.*'))


if __name__ == '__main__':
  absltest.main()
