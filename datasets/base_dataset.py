import numpy as np
from abc import ABC, abstractmethod
import math


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent: float = train_set_percent
        self.valid_set_percent: float = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        ...

    @property
    @abstractmethod
    def inputs(self):
        ...

    def _divide_into_sets(self):
        train_delimeter = math.floor(self.inputs.shape[0] * self.train_set_percent)
        valid_delimeter = train_delimeter + math.floor(self.inputs.shape[0] * self.valid_set_percent)
        self.inputs_train = self.inputs[:train_delimeter]
        self.targets_train = self.targets[:train_delimeter]
        self.inputs_valid = self.inputs[train_delimeter:valid_delimeter]
        self.targets_valid = self.targets[train_delimeter:valid_delimeter]
        self.inputs_test = self.inputs[valid_delimeter:]
        self.targets_test = self.targets[valid_delimeter:]
