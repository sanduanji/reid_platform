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




def cr_softmaxmodel(arch, features, dropout, num_classes):
    # Create model
    model = models.create(arch, num_features=features, dropout=dropout, num_classes=num_classes)
    return model


def cr_tripletmodel(arch, dropout, features):
    model = models.create(arch, num_features=1024, dropout=dropout, num_classes=features)
    return model


def cr_oimmodel(arch, features, dropout ):
    model = models.create(arch, num_features=features, norm=True, dropout=dropout)
    return model
