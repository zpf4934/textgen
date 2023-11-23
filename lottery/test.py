# -*- coding:utf-8 -*-
"""
# File       : test.py
# Time       ：2023/11/1 11:13
# Author     ：andy
# version    ：python 3.9
"""
import pycaret
from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment

data = get_data('./red')
exp = ClassificationExperiment()
exp.setup(data, target='label', session_id=123, log_experiment='wandb', experiment_name='class')
exp.compare_models(exclude=['lightgbm'])
