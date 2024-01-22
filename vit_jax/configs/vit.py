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

r"""Fine-tunes a Vision Transformer.

Example for fine-tuning a ViT-B/16 on CIFAR10:

python -m vit_jax.main --workdir=/tmp/vit \
    --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
    --config.pretrained_dir='gs://vit_models/imagenet21k'
"""

from vit_jax.configs import common
from vit_jax.configs import models


def get_config(model_dataset):
  """Returns default parameters for finetuning ViT `model` on `dataset`."""
  model, dataset = model_dataset.split(',')
  config = common.with_dataset(common.get_config(), dataset)
  get_model_config = getattr(models, f'get_{model}_config')
  config.model = get_model_config()

  if model == 'b16' and dataset == 'cifar10':
    config.base_lr = 0.01

  return config
