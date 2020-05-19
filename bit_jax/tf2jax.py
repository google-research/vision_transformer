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

import numpy as np
import re


def transform_params(params, params_tf, num_classes):
  # BiT and JAX models have different naming conventions, so we need to
  # properly map TF weights to JAX weights
  params['root_block']['conv_root']['kernel'] = (
    params_tf['resnet/root_block/standardized_conv2d/kernel'])

  for block in ['block1', 'block2', 'block3', 'block4']:
    units = set([re.findall(r'unit\d+', p)[0] for p in params_tf.keys()
                 if p.find(block) >= 0])
    for unit in units:
      for i, group in enumerate(['a', 'b', 'c']):
        params[block][unit][f'conv{i+1}']['kernel'] = (
          params_tf[f'resnet/{block}/{unit}/{group}/'
                    'standardized_conv2d/kernel'])
        params[block][unit][f'gn{i+1}']['bias'] = (
          params_tf[f'resnet/{block}/{unit}/{group}/'
                    'group_norm/beta'][None, None, None])
        params[block][unit][f'gn{i+1}']['scale'] = (
          params_tf[f'resnet/{block}/{unit}/{group}/'
                    'group_norm/gamma'][None, None, None])

      projs = [p for p in params_tf.keys()
               if p.find(f'{block}/{unit}/a/proj') >= 0]
      assert len(projs) <= 1
      if projs:
        params[block][unit]['conv_proj']['kernel'] = params_tf[projs[0]]

  params['norm-pre-head']['bias'] = (
    params_tf['resnet/group_norm/beta'][None, None, None])
  params['norm-pre-head']['scale'] = (
    params_tf['resnet/group_norm/gamma'][None, None, None])

  params['conv_head']['kernel'] = np.zeros(
    (params['conv_head']['kernel'].shape[0], num_classes), dtype=np.float32)
  params['conv_head']['bias'] = np.zeros(num_classes, dtype=np.float32)

