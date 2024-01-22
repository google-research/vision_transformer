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

r"""Fine-tunes a Vision Transformer / Hybrid from AugReg checkpoint.

Example for fine-tuning a R+Ti/16 on cifar100:

python -m vit_jax.main --workdir=/tmp/vit \
    --config=$(pwd)/vit_jax/configs/augreg.py:R_Ti_16 \
    --config.dataset=oxford_iiit_pet \
    --config.pp.train='train[:90%]' \
    --config.base_lr=0.01

Note that by default, the best i21k pre-trained checkpoint by upstream
validation accuracy is chosen. You can also manually select a model by
specifying the full name (without ".npz" extension):

python -m vit_jax.main --workdir=/tmp/vit \
    --config=$(pwd)/vit_jax/configs/augreg.py:R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0 \
    --config.dataset=oxford_iiit_pet \
    --config.pp.train='train[:90%]' \
    --config.base_lr=0.01
"""

import ml_collections

from vit_jax.configs import common
from vit_jax.configs import models


def get_config(model_or_filename):
  """Returns default parameters for finetuning ViT `model` on `dataset`."""
  config = common.get_config()

  config.pretrained_dir = 'gs://vit_models/augreg'

  config.model_or_filename = model_or_filename
  model = model_or_filename.split('-')[0]
  if model not in models.AUGREG_CONFIGS:
    raise ValueError(f'Unknown Augreg model "{model}"'
                     f'- not found in {set(models.AUGREG_CONFIGS.keys())}')
  config.model = models.AUGREG_CONFIGS[model].copy_and_resolve_references()
  config.model.transformer.dropout_rate = 0  # No AugReg during fine-tuning.

  # These values are often overridden on the command line.
  config.base_lr = 0.03
  config.total_steps = 500
  config.warmup_steps = 100
  config.pp = ml_collections.ConfigDict()
  config.pp.train = 'train'
  config.pp.test = 'test'
  config.pp.resize = 448
  config.pp.crop = 384

  # This value MUST be overridden on the command line.
  config.dataset = ''

  return config
