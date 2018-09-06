from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

from reid import datasets
from reid import models
from reid.loss import TripletLoss
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.loss import OIMLoss


from examples.pl_createmodel import cr_oimmodel, cr_softmaxmodel, cr_tripletmodel
from examples.pl_parameters import *



def softmax_op(model, lr, momentum, weight_decay, ):
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr = lr,
                                momentum = momentum,
                                weight_decay = weight_decay,
                                nesterov = True)
    return optimizer


def triplet_op(model, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr,
                                 weight_decay = weight_decay)
    return optimizer


def oim_op(model, lr, momentum, weight_decay):
    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr = lr,
                                momentum = momentum,
                                weight_decay = weight_decay,
                                nesterov = True)
    return optimizer