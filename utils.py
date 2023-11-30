import torch
from collections import OrderedDict, defaultdict
from enum import Enum

def l2_between_dicts(dict_1, dict_2, normalize=False):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    dict_1_tensor = torch.cat(tuple([t.view(-1) for t in dict_1_values]))
    dict_2_tensor = torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    if normalize:
        dict_1_tensor = (dict_1_tensor-dict_1_tensor.mean().item()) / (dict_1_tensor.std().item() + 1e-6)
        dict_2_tensor = (dict_2_tensor-dict_2_tensor.mean().item()) / (dict_2_tensor.std().item() + 1e-6)
        dict_2_tensor = dict_2_tensor.detach()
    return (dict_1_tensor-dict_2_tensor).pow(2).mean()

def _get_grads(optimizer, featurizer, loss):
    optimizer.zero_grad()
    loss.backward(retain_graph=True, create_graph=True)
    dict = OrderedDict(
        [
            (name, weights.grad.clone().view(weights.grad.size(0), -1))
            for name, weights in featurizer.named_parameters()
        ]
    )

    return dict


def get_file_name(file_name):
    if type(file_name) is str:
        file_short_name, file_extension = os.path.splitext(file_name)
        return file_short_name
    return None

class ProcessingType(Enum):
    TRAIN = "train"
    TEST = "test"
