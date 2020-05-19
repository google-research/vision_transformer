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

# Lint as: python3
"""Various utilities from my toolbox at github.com/lucasb-eyer/lbtoolbox."""

import collections
import json
import signal
import time

import numpy as np


class Uninterrupt:
  """Context manager to gracefully handle interrupts.

  Use as:
  with Uninterrupt() as u:
      while not u.interrupted:
          # train
  """

  def __init__(self, sigs=(signal.SIGINT, signal.SIGTERM), verbose=False):
    self.sigs = sigs
    self.verbose = verbose
    self.interrupted = False
    self.orig_handlers = None

  def __enter__(self):
    if self.orig_handlers is not None:
      raise ValueError("Can only enter `Uninterrupt` once!")

    self.interrupted = False
    self.orig_handlers = [signal.getsignal(sig) for sig in self.sigs]

    def handler(signum, frame):
      del signum  # unused
      del frame  # unused
      self.release()
      self.interrupted = True
      if self.verbose:
        print("Interruption scheduled...", flush=True)

    for sig in self.sigs:
      signal.signal(sig, handler)

    return self

  def __exit__(self, type_, value, tb):
    self.release()

  def release(self):
    if self.orig_handlers is not None:
      for sig, orig in zip(self.sigs, self.orig_handlers):
        signal.signal(sig, orig)
      self.orig_handlers = None


class Timer:
  """Context timing its scope."""

  def __init__(self, donecb):
    self.cb = donecb

  def __enter__(self):
    self.t0 = time.time()

  def __exit__(self, exc_type, exc_value, traceback):
    t = time.time() - self.t0
    self.cb(t)


class Chrono:
  """Chronometer for poor-man's (but convenient!) profiling."""

  def __init__(self):
    self.timings = collections.OrderedDict()

  def measure(self, what):
    return Timer(lambda t: self._done(what, t))

  def _done(self, what, t):
    self.timings.setdefault(what, []).append(t)

  def times(self, what):
    return self.timings[what]

  def avgtime(self, what, dropfirst=False):
    timings = self.timings[what]
    if dropfirst and len(timings) > 1:
      timings = timings[1:]
    return sum(timings)/len(timings)

  def __str__(self, fmt="{:{w}.5f}", dropfirst=False):
    avgtimes = {k: self.avgtime(k, dropfirst) for k in self.timings}
    l = max(map(len, avgtimes))
    w = max(len(fmt.format(v, w=0)) for v in avgtimes.values())
    avg_by_time = sorted(avgtimes.items(), key=lambda t: t[1], reverse=True)
    return "\n".join(f"{name:{l}s}: " + fmt.format(t, w=w) + "s"
                     for name, t in avg_by_time)


def create_dat(basename, dtype, shape, fillvalue=None, **meta):
  """Creates mem-mapped numpy array plus metadata.

  Creates a data file at `basename` and returns a writeable mem-map backed
  numpy array to it. Can also be passed any json-serializable keys and values
  in `meta`.
  """
  xm = np.memmap(basename, mode="w+", dtype=dtype, shape=shape)
  xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=xm)
  # xa.flush = xm.flush  # Sadly, we can't just add attributes to a numpy array, need to subclass it.

  if fillvalue is not None:
    xa.fill(fillvalue)
    # xa.flush()
    xm.flush()

  meta.setdefault("dtype", np.dtype(dtype).str)
  meta.setdefault("shape", shape)
  json.dump(meta, open(basename + ".json", "w+"))

  return xa


def load_dat(basename, mode="r"):
  """Loads file created via `create_dat` as mem-mapped numpy array.

  Returns a read-only mem-mapped numpy array to file at `basename`.
  If `mode` is set to `'r+'`, the data can be written, too.
  """
  desc = json.load(open(basename + ".json", "r"))
  dtype, shape = desc["dtype"], desc["shape"]
  xm = np.memmap(basename, mode=mode, dtype=dtype, shape=shape)
  xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=xm)
  # xa.flush = xm.flush  # Sadly, we can't just add attributes to a numpy array, need to subclass it.
  return xa
