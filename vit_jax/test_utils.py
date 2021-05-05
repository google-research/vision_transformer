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

import jax
import jax.numpy as jnp

from vit_jax import checkpoint
from vit_jax import models


def create_checkpoint(model_config, path):
  """Initializes model and stores weights in specified path."""
  model = models.VisionTransformer(num_classes=1, **model_config)
  variables = model.init(
      jax.random.PRNGKey(0),
      jnp.ones([1, 16, 16, 3], jnp.float32),
      train=False,
  )
  checkpoint.save(variables['params'], path)
