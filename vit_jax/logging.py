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

import logging
import os
import logging.config


def setup_logger(log_dir):
  """Creates and returns a fancy logger."""
  # Why is setting up proper logging so !@?#! ugly?
  os.makedirs(log_dir, exist_ok=True)
  logging.config.dictConfig({
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {
          'standard': {
              'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
          },
      },
      'handlers': {
          'stderr': {
              'level': 'INFO',
              'formatter': 'standard',
              'class': 'logging.StreamHandler',
              'stream': 'ext://sys.stderr',
          },
          'logfile': {
              'level': 'DEBUG',
              'formatter': 'standard',
              'class': 'logging.FileHandler',
              'filename': os.path.join(log_dir, 'train.log'),
              'mode': 'a',
          }
      },
      'loggers': {
          '': {
              'handlers': ['stderr', 'logfile'],
              'level': 'DEBUG',
              'propagate': True
          },
      }
  })
  logger = logging.getLogger(__name__)
  logger.flush = lambda: [h.flush() for h in logger.handlers]
  return logger
