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

from vit_jax import models_lit
from vit_jax import models_mixer
from vit_jax import models_vit
from vit_jax.configs import models as model_configs

# Note that you probably want to import the individual modules separately
# instead (e.g. not depending on tensorflow_text required by models_lit if
# you're only interested in image models).
AddPositionEmbs = models_vit.AddPositionEmbs
MlpBlock = models_vit.MlpBlock
Encoder1DBlock = models_vit.Encoder1DBlock
Encoder = models_vit.Encoder

LitModel = models_lit.LitModel
MlpMixer = models_mixer.MlpMixer
VisionTransformer = models_vit.VisionTransformer


def get_model(name, **kw):
  """Returns a model as specified in `model_configs.MODEL_CONFIGS`."""
  if name.startswith('Mixer-'):
    return MlpMixer(**model_configs.MODEL_CONFIGS[name], **kw)
  elif name.startswith('LiT-'):
    return LitModel(**model_configs.MODEL_CONFIGS[name], **kw)
  else:
    return VisionTransformer(**model_configs.MODEL_CONFIGS[name], **kw)
