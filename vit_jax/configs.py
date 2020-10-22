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

import ml_collections


def get_testing():
  """Returns a minimal configuration for testing."""
  config = ml_collections.ConfigDict()
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 1
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 1
  config.transformer.num_heads = 1
  config.transformer.num_layers = 1
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


def get_b16_config():
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


def get_b32_config():
  """Returns the ViT-B/32 configuration."""
  config = get_b16_config()
  config.patches.size = (32, 32)
  return config


def get_l16_config():
  """Returns the ViT-L/16 configuration."""
  config = ml_collections.ConfigDict()
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 1024
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 4096
  config.transformer.num_heads = 16
  config.transformer.num_layers = 24
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


def get_l32_config():
  """Returns the ViT-L/32 configuration."""
  config = get_l16_config()
  config.patches.size = (32, 32)
  return config
