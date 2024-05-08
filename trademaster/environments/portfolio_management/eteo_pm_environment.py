from __future__ import annotations
import torch
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import numpy as np
from trademaster.utils import get_attr, print_metrics
import pandas as pd
from ..custom import Environments
from ..builder import ENVIRONMENTS
from trademaster.pretrained import pretrained
from gym import spaces
from collections import OrderedDict
import pickle
import os.path as osp

@ENVIRONMENTS.register_module()
class PortfolioManagementETEOEnvironment(Environments):
    def __init__(self, config):
        super(PortfolioManagementETEOEnvironment, self).__init__()
        self.dataset = get_attr(config, "dataset", None)
        self.task = get_attr(config, "task", "train")
        
        