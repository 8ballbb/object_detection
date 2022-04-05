import re
from collections.abc import Iterable


def update_config(
    cfg_file, labels, height=None, width=None, max_batch=None, steps=None, batch=None, subdivisions=None):
  with open(cfg_file) as f:
    s = f.read()
  # Height and Width
  if height is not None:
    assert height % 32 == 0, "height must be divisible by 32 (default 416)"
    s = re.sub("height=\d*", f"height={height}", s)
  if width is not None:
    assert width % 32 == 0, "width must be divisible by 32 (default 416)"
    s = re.sub("width=\d*", f"width={width}", s)
  
  # Max Batch
  if max_batch is not None:
    assert max_batch >= len(labels) * 2000, "Minimum 2000 per class (default 500,200)"
  else:
    max_batch = len(labels) * 2000
  s = re.sub("max_batches = \d*", f"max_batches = {max_batch}", s)

  # Steps
  if steps is not None:
    assert isinstance(steps, Iterable) and len(steps) == 2, "Expected steps to be an iterable and length of 2 (default 400000,450000)"
  else:
    step1, step2 = 0.8 * max_batch, 0.9 * max_batch
  s = re.sub("steps=\d*,\d*", f"steps={int(step1)},{int(step2)}", s)
  
  # Num Filters
  num_filters = (len(labels) + 5) * 3
  s = re.sub("classes=\d*", f"classes={len(labels)}", s)
  s = re.sub("pad=1\nfilters=\d*", f"pad=1\nfilters={int(num_filters)}", s)
  # Batch and subdivisions
  if batch is not None:
    s = re.sub("batch=\d*", f"batch={batch}", s)
  if subdivisions is not None:
    if batch is None:
      batch = int(re.findall("\d+", re.findall("batch=\d*", s)[1])[0])
    assert batch % subdivisions == 0, f"batch ({batch}) must be divisible by subdivision ({subdivisions})"
    s = re.sub("subdivisions=\d*", f"subdivisions={subdivisions}", s)
  
  with open(cfg_file, "w") as f:
    f.write(s)
