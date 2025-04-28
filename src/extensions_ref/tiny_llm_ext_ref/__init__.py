# Copyright © 2023 Apple Inc.

import mlx.core as mx

from ._ext import *
from pathlib import Path

current_path = Path(__file__).parent
load_library(mx.gpu, str(current_path))
