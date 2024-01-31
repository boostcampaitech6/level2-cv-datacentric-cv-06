# basic random seed
import os
import random

import numpy as np

# torch random seed
import torch

DEFAULT_RANDOM_SEED = 666


def set_python(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)  # python 내장함수 random에 seed를 추가합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)  # python hash에 seed를 추가합니다.
    np.random.seed(seed)  # numpy library에 seed를 추가합니다.


def set_torch(seed=DEFAULT_RANDOM_SEED):  # pytorch를 위한 seed를 추가합니다.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setting_seed(seed=DEFAULT_RANDOM_SEED): 
    set_python(seed)
    set_torch(seed)


def _init_fn(worker_id):  # DataLoader에 seed를 추가하기 위한 함수입니다.
    np.random.seed(
        int(DEFAULT_RANDOM_SEED + worker_id)
    )  # default seed에 worker id를 더하여, 2개 이상의 worker에도 각각 seed를 부여하고, data loading이 재현가능하도록 합니다.