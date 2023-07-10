import os
import torch
import torch.nn as nn

from logger import variable_logger as logger
log = logger(is_active=True)
log.is_active = False

class BaseNN(nn.Module):
    """
    Base NN, all other NN will base on this
    - __init__: virtual
    - forward(): virtual
    """

    def __init__(self):
        super().__init__()

    def forward(self, i: torch.Tensor): return i

    # Below are some save / load functions. You could use them for single model save / load,
    # but if you're working with a more complex environment, like with multiple models or
    # you want to save / load the optimizer, scheduler, etc. You should use the save_all() / load_all()
    # functions in logger.py
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    # The auto_save() functions take epoch and iter as params, and auto_load() will load the ckpts
    # with the newest epoch and iter. Notice that the ckpts are in the form of "name-epoch-{epoch}-iter-{iter}.pth"
    def auto_save(self, prefix:str, epoch: int, iter: int): # name-epoch-{epoch}-iter-{iter}.pth
        save_name = prefix + "-epoch-" + str(epoch) + "-iter-" + str(iter) + ".pth"
        self.save(save_name)
    def auto_load(self, ckpt_dir: str, prefix: str) -> tuple[int, int]:
        self.ckpts = os.listdir(ckpt_dir)
        all_ckpt = []
        for ckpt_name in self.ckpts:
            ckpt_name = ckpt_name[:-4].split("-") # remove ".pth" and split by "-"
            if(ckpt_name[0] == prefix):
                all_ckpt.append( (ckpt_name, int(ckpt_name[2]), int(ckpt_name[4])) ) # name-epoch-{epoch}-iter-{iter}.pth
        if (all_ckpt == []):
            print(f"No checkpoint founded in '{ckpt_dir}' with name '{prefix}'.")
            return 0, 0
        res = sorted(all_ckpt, key = lambda x: (x[1], x[2]))
        newest = '-'.join(res[-1][0]) + ".pth"
        print("loading newest checkpoints " + os.path.join(ckpt_dir, newest))
        self.load(os.path.join(ckpt_dir, newest))
        return int(res[-1][1]), int(res[-1][2]) # start_epoch, start_iter

class some_layer(BaseNN):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, i):
        return i
