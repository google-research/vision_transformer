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

import ml_collections


def get_config():
  """Returns a configuration for inference_time.py."""
  config = ml_collections.ConfigDict()

  # Which model to use -- see ./models.py
  config.model_name = 'ViT-B_32'
  # Where to store training logs.
  config.log_dir = '.'

  # Number of steps to measure.
  config.steps = 30
  # Number of steps before measuring.
  config.initial_steps = 10

  # Batch size
  config.batch = 0
  # Number of output classes.
  config.num_classes = 0
  # Image size (width=height).
  config.image_size = 0

  config.train = 'inference_time'

  return config
