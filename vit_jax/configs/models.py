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

# The key of this dictionary refers to basename in the directory:
# https://console.cloud.google.com/storage/vit_models/
# Note that some names (e.g. "testing", but also some models only available in
# the AugReg paper) are not actually present in that directory.
MODEL_CONFIGS = {}

# The key of this dictionary refers to the first part (delimited by "-") of the
# filename of the checkpoint in:
# https://console.cloud.google.com/storage/vit_models/augreg/index.csv
AUGREG_CONFIGS = {}


def _register(get_config):
  """Adds reference to model config into MODEL_CONFIGS and AUGREG_CONFIGS."""
  config = get_config().lock()
  name = config.get('model_name')
  MODEL_CONFIGS[name] = config
  if 'Mixer' not in name and name not in ('testing', 'ViT-L_32', 'R50+ViT-B_16',
                                          'ViT-H_14'):
    # Note: we're using stricter filenames for AugReg checkpoints so they can be
    # used both as filesystem filenames and URIs without escaping.
    augreg_name = name.replace('ViT-', '').replace('+', '_')
    AUGREG_CONFIGS[augreg_name] = config
  return get_config


@_register
def get_testing_config():
  """Returns a simple config used for testing."""
  config = ml_collections.ConfigDict()
  # Only used for testing.
  config.model_name = 'testing'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 10
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 10
  config.transformer.num_heads = 2
  config.transformer.num_layers = 1
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_testing_unpooled_config():
  """Returns a simple config used for testing unpooled version."""
  config = get_testing_config()
  # Only used for testing.
  config.model_name = 'testing-unpooled'
  config.classifier = 'unpooled'
  return config


# ViT-X/16 & ViT-H/14
#####################


@_register
def get_ti16_config():
  """Returns the ViT-Ti/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-Ti_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 192
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_s16_config():
  """Returns the ViT-S/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-S_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 384
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 1536
  config.transformer.num_heads = 6
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_b16_config():
  """Returns the ViT-B/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-B_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.num_layers = 12
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_l16_config():
  """Returns the ViT-L/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-L_16'
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


@_register
def get_h14_config():
  """Returns the ViT-H/14 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'ViT-H_14'
  config.patches = ml_collections.ConfigDict({'size': (14, 14)})
  config.hidden_size = 1280
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 5120
  config.transformer.num_heads = 16
  config.transformer.num_layers = 32
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.1
  config.classifier = 'token'
  config.representation_size = None
  return config


@_register
def get_s16_gap_norep_config():
  """Returns ViT-S/16 with classifier=gap, representation=None."""
  config = get_s16_config()
  config.model_name = 'ViT-S_16-gap-norep'
  config.classifier = 'gap'
  config.representation_size = None
  return config


@_register
def get_b16_gap_norep_config():
  """Returns ViT-B/16 with classifier=gap, representation=None."""
  config = get_b16_config()
  config.model_name = 'ViT-B_16-gap-norep'
  config.classifier = 'gap'
  config.representation_size = None
  return config


# ViT-X/8
#########


@_register
def get_b8_config():
  """Returns the ViT-B/8 configuration."""
  config = get_b16_config()
  config.model_name = 'ViT-B_8'
  config.patches.size = (8, 8)
  return config


# ViT-X/32
##########


@_register
def get_s32_config():
  """Returns the ViT-S/32 configuration."""
  config = get_s16_config()
  config.model_name = 'ViT-S_32'
  config.patches.size = (32, 32)
  return config


@_register
def get_b32_config():
  """Returns the ViT-B/32 configuration."""
  config = get_b16_config()
  config.model_name = 'ViT-B_32'
  config.patches.size = (32, 32)
  return config


@_register
def get_l32_config():
  """Returns the ViT-L/32 configuration."""
  config = get_l16_config()
  config.transformer.dropout_rate = 0.0
  config.model_name = 'ViT-L_32'
  config.patches.size = (32, 32)
  return config



@_register
def get_s32_gap_norep_config():
  """Returns ViT-S/32 with classifier=gap, representation=None."""
  config = get_s32_config()
  config.model_name = 'ViT-S_32-gap-norep'
  config.classifier = 'gap'
  config.representation_size = None
  return config


@_register
def get_b32_gap_norep_config():
  """Returns ViT-B/32 with classifier=gap, representation=None."""
  config = get_b32_config()
  config.model_name = 'ViT-B_32-gap-norep'
  config.classifier = 'gap'
  config.representation_size = None
  return config


# Hybrids R+ViT-X/16
####################


@_register
def get_r_ti16_config():
  """Returns the Resnet stem + ViT-Ti/16 configuration."""
  config = get_ti16_config()
  config.model_name = 'R+ViT-Ti_16'
  config.patches.size = (8, 8)
  config.resnet = ml_collections.ConfigDict()
  # The resnet stem alone downscales 2x, making /16 with 8x8 patches.
  config.resnet.num_layers = ()
  config.resnet.width_factor = 1
  return config


@_register
def get_r50_b16_config():
  """Returns the Resnet50 + ViT-B/16 configuration."""
  config = get_b16_config()
  config.transformer.dropout_rate = 0.1
  config.model_name = 'R50+ViT-B_16'
  config.patches.size = (1, 1)
  config.resnet = ml_collections.ConfigDict()
  # Note that the "real" Resnet50 has (3, 4, 6, 3) bottleneck blocks. Here
  # we're using (3, 4, 9) configuration so we get a downscaling of 2^(1 + 3)=16
  # which results in an effective patch size of /16.
  config.resnet.num_layers = (3, 4, 9)
  config.resnet.width_factor = 1
  return config


# Hybrids R+ViT-X/32
####################


@_register
def get_r26_b32_config():
  """Returns the Resnet26 + ViT-B/32 configuration."""
  config = get_b32_config()
  config.model_name = 'R26+ViT-B_32'
  config.patches.size = (1, 1)
  config.resnet = ml_collections.ConfigDict()
  # Using four bottleneck blocks results in a downscaling of 2^(1 + 4)=32 which
  # results in an effective patch size of /32.
  config.resnet.num_layers = (2, 2, 2, 2)
  config.resnet.width_factor = 1
  return config


@_register
def get_r26_s32_config():
  """Returns the Resnet26 + ViT-S/32 configuration."""
  config = get_s16_config()
  config.model_name = 'R26+ViT-S_32'
  config.patches.size = (1, 1)
  config.resnet = ml_collections.ConfigDict()
  # Using four bottleneck blocks results in a downscaling of 2^(1 + 4)=32 which
  # results in an effective patch size of /32.
  config.resnet.num_layers = (2, 2, 2, 2)
  config.resnet.width_factor = 1
  return config


@_register
def get_r50_l32_config():
  """Returns the Resnet50 + ViT-L/32 configuration."""
  config = get_l16_config()
  config.model_name = 'R50+ViT-L_32'
  config.patches.size = (1, 1)
  config.resnet = ml_collections.ConfigDict()
  # Using four bottleneck blocks results in a downscaling of 2^(1 + 4)=32 which
  # results in an effective patch size of /32.
  config.resnet.num_layers = (3, 4, 6, 3)
  config.resnet.width_factor = 1
  return config


# Mixers
########


@_register
def get_mixer_b16_config():
  """Returns Mixer-B/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'Mixer-B_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_dim = 768
  config.num_blocks = 12
  config.tokens_mlp_dim = 384
  config.channels_mlp_dim = 3072
  return config


@_register
def get_mixer_b32_config():
  """Returns Mixer-B/32 configuration."""
  config = get_mixer_b16_config()
  config.model_name = 'Mixer-B_32'
  config.patches = ml_collections.ConfigDict({'size': (32, 32)})
  return config


@_register
def get_mixer_l16_config():
  """Returns Mixer-L/16 configuration."""
  config = ml_collections.ConfigDict()
  config.model_name = 'Mixer-L_16'
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_dim = 1024
  config.num_blocks = 24
  config.tokens_mlp_dim = 512
  config.channels_mlp_dim = 4096
  return config


# LiT
#####


@_register
def get_lit_b16b_config():
  """Returns a LiT model with ViT-Base and BERT-Base towers."""
  config = ml_collections.ConfigDict()
  config.model_name = 'LiT-B16B'
  config.out_dim = (768, 768)
  config.image = get_b16_config()
  config.text_model = 'bert'
  config.text = {}
  config.text.config = 'base'
  config.pp = {}
  config.pp.tokenizer_name = 'bert'
  config.pp.size = 224
  config.pp.max_len = 16
  return config


@_register
def get_lit_b16b_2_config():
  """Returns an improved LiT model with ViT-Base and BERT-Base towers."""
  config = get_lit_b16b_config()
  config.model_name = 'LiT-B16B_2'
  config.out_dim = (None, 768)
  return config


@_register
def get_lit_l16l_config():
  """Returns a LiT model with ViT-Large and BERT-Large towers."""
  config = ml_collections.ConfigDict()
  config.model_name = 'LiT-L16L'
  config.out_dim = (None, 1024)
  config.image = get_l16_config()
  config.text_model = 'bert'
  config.text = {}
  config.text.config = 'large'
  config.pp = {}
  config.pp.tokenizer_name = 'bert'
  config.pp.size = 224
  config.pp.max_len = 16
  return config


@_register
def get_lit_l16s_config():
  """Returns a LiT model with ViT-Large and small text towers."""
  config = ml_collections.ConfigDict()
  config.model_name = 'LiT-L16S'
  config.out_dim = (None, 1024)
  config.image = get_l16_config()
  config.text_model = 'text_transformer'
  config.text = {}
  config.text.width = 384
  config.text.num_layers = 12
  config.text.mlp_dim = 1536
  config.text.num_heads = 6
  config.text.vocab_size = 16_000
  config.pp = {}
  config.pp.tokenizer_name = 'sentencepiece'
  config.pp.size = 224
  config.pp.max_len = 16
  return config


@_register
def get_lit_l16ti_config():
  """Returns a LiT model with ViT-Large and tiny text towers."""
  config = ml_collections.ConfigDict()
  config.model_name = 'LiT-L16Ti'
  config.out_dim = (None, 1024)
  config.image = get_l16_config()
  config.text_model = 'text_transformer'
  config.text = {}
  config.text.width = 192
  config.text.num_layers = 12
  config.text.mlp_dim = 768
  config.text.num_heads = 3
  config.text.vocab_size = 16_000
  config.pp = {}
  config.pp.tokenizer_name = 'sentencepiece'
  config.pp.size = 224
  config.pp.max_len = 16
  return config
