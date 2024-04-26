import dataclasses
import enum

import numpy


@dataclasses.dataclass
class Metrics(dict):
    explained_variance: numpy.array = dataclasses.field(default_factory=lambda: numpy.array([]))
    entropy: numpy.array = dataclasses.field(default_factory=lambda: numpy.array([]))
    policy_loss: numpy.array = dataclasses.field(default_factory=lambda: numpy.array([]))
    value_loss: numpy.array = dataclasses.field(default_factory=lambda: numpy.array([]))
    loss: numpy.array = dataclasses.field(default_factory=lambda: numpy.array([]))
