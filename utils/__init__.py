import torch
from torchsummary import summary

SEED = 42
DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print("Device Selected:", DEVICE)
    return DEVICE


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if get_device() == 'cuda':
        torch.cuda.manual_seed(seed)

def model_summary(model, input_size):
    return summary(model, input_size=input_size)