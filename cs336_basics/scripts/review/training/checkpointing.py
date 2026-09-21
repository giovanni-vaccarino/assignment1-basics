import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    obj = {}
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    obj["iteration"] = iteration
    torch.save(obj, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"]) # we could do some checks to ensure the object is not empty ...
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]
