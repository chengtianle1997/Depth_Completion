"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from .model import uncertainty_net as model
from .model import global_net as erfnet
from .model import local_net as hourglass

model_dict = {'mod': model, 'erfnet': erfnet, 'hourglass': hourglass}

def allowed_models():
    return model_dict.keys()


def define_model(mod, **kwargs):
    if mod not in allowed_models():
        raise KeyError("The requested model: {} is not implemented".format(mod))
    else:
        return model_dict[mod](**kwargs)
