from colored import fg, attr
from inspect import currentframe, getframeinfo
import inspect

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from torch.utils.tensorboard.writer import SummaryWriter


class logger():
    def __init__(self, path):
        self.writer = SummaryWriter(log_dir=path)

    def log(self, name: str, loss: float, n_iter: int):
        self.writer.add_scalar(name, loss, global_step=n_iter)

    def log_image(self, name: str, image: torch.Tensor, n_iter: int, img_formats="CHW"):
        self.writer.add_image(name, np.array(
            image.detach().cpu()), dataformats=img_formats, global_step=n_iter)

    def log_image_batched(self, name: str, images: torch.Tensor, n_iter: int, ncol: int = 2, img_formats="CHW"):
        grid = torchvision.utils.make_grid(images, nrow=ncol)
        self.writer.add_image(
            name, grid, global_step=n_iter, dataformats=img_formats)
        self.writer.close()

    def log_image_list(self, name: str, images: list[torch.Tensor], n_iter: int, ncol: int = 2, img_formats="CHW"):
        grid = torchvision.utils.make_grid(images, nrow=ncol)
        self.writer.add_image(
            name, grid, global_step=n_iter, dataformats=img_formats)
        self.writer.close()

    def log_text_batched(self, name: str, text: str, n_iter: int):
        self.writer.add_text(name, text, global_step=n_iter)

    def close(self):
        self.writer.close()


class variable_logger():
    def __init__(self, is_active=True):
        self.book = {}
        self.is_active = is_active

    def log(self, name: str, value, frameinfo: inspect.Traceback=None, is_once: bool = False):
        if not self.is_active:
            return
        if is_once == True and name in self.book.keys():
            return
        if name not in self.book.keys():
            self.book[name] = value
        if (frameinfo == None):
            print(f"{fg(22)}logger: {attr(0)}{fg(42)}{name}{attr(0)}\t{fg(44)}{value}{attr(0)}")
        else:
            filename = "/".join(frameinfo.filename.split("/")[-2:])
            term_width = os.get_terminal_size().columns

            left = f"{fg(22)}logger: {fg(42)}{name}: {fg(44)}{value}{attr(0)}"
            uncolored_left = f"logger: {name}: {value}"
            right = f"{fg(22)}from {fg(43)}{filename} {fg(26)}in line {fg(45)}{frameinfo.lineno}{attr(0)};"
            uncolored_right = f"from {filename} in line {frameinfo.lineno};"

            left_size = term_width - len(uncolored_right)

            if (left_size < 0):
                right = f"{fg(43)}{filename}{attr(0)}.ln {fg(45)}{frameinfo.lineno}{attr(0)};"
                left_size = max(term_width - len(uncolored_right), 0)

            print(f"{left:<{left_size}}{right}")

            # print("*"*left_size)
            # print(left)
            # print("*"*len(uncolored_left))
            # print(right)
            # print("*"*len(uncolored_right))
            # exit()

    def log_all(self):
        print("logger: ", end="")
        for i in self.book.keys():
            print(i, "- ", f"{self.book[i]:.5f}", sep="", end="; ")
        print("\n", end="")

    def update(self, name: str, value):
        self.book[name] = value


def save_all(path:str, epoch=None, n_iter=None, **datas):
    """ 
    datas must be in a form of dict["models", "variables"], where:
    - models:dict here includes the model, optimizer, scheduler, etc.
    - variables:dict here includes any variables, like string or int.
    if epoch and n_iter are not None, they're adding to variables automatically.
    e.g. save_all("./ckpts", epoch=1, n_iter=0,
        models={"encoder":enc, "optimizer":optimizer})
    """
    ckpt = {}
    keys = datas.keys()
    for i in datas["models"].keys():
        ckpt[i] = datas["models"][i].state_dict()
    if "variables" in keys:
        for i in datas["variables"].keys():
            ckpt[i] = datas["variables"][i]
    if epoch != None and n_iter != None:
        ckpt["epoch"] = epoch
        ckpt["n_iter"] = n_iter
        name = f"total-epoch-{epoch}-iter-{n_iter}.pth"
        torch.save(ckpt, os.path.join(path, name))
    else:
        torch.save(ckpt, path)

def load_all(path:str, epoch=None, n_iter=None):
    """ 
    datas will be returned in a form of dict["models", "variables"], where:
    - models:dict here includes the model, optimizer, scheduler, etc.
    - variables:dict here includes any variables, like string or int.
    this function ONLY load the ckpts which ** have a prefix of "total-" **
    e.g. 
    datas = load_all("./ckpts/epoch-0-iter-100", datas)

    if epoch & n_iter are both not None (int stead), and "path" is not a file,
        then if "path" is a file, load "path". else
        load "path/total-epoch-iter-n_iter.pth".
    else:
        load "path", if it's a file. else:
        try to load "path/total-epoch-{epoch}-iter-{n_iter}.pth", where we load the newest one.
        if failed, then throw a exception.
    """
    if (epoch != None and n_iter != None):
        if (os.path.isfile(path)):
            datas = torch.load(path)
            print("Load from", path+",", "Even if EPOCH and N_ITER are given.")
        else:
            datas = torch.load(os.path.join(path, f"total-epoch-{epoch}-iter-{n_iter}.pth"))
            print("Load from", os.path.join(path, f"total-epoch-{epoch}-iter-{n_iter}.pth")+".")
    else:
        if (os.path.isfile(path)):
            datas = torch.load(path)
            print("Load from", path+".")
        else: # get the newest one by soring
            ckpts = os.listdir(path)
            if (ckpts == []):
                print(f"No checkpoint founded in '{path}' with name 'total'. return None.")
                datas = None
                # raise RuntimeError(f"No checkpoint founded in '{path}' with name 'total' and {path} is not a ckpt file.")
            else:
                all_ckpt = []
                for ckpt_name in ckpts:
                    ckpt_name = ckpt_name[:-4].split("-") # remove ".pth" and split by "-"
                    if(ckpt_name[0] == "total"):
                        all_ckpt.append( (ckpt_name, int(ckpt_name[2]), int(ckpt_name[4])) ) # name-epoch-{epoch}-iter-{iter}.pth

                res = sorted(all_ckpt, key = lambda x: (x[1], x[2]))
                newest = '-'.join(res[-1][0]) + ".pth"
                print("Load from NEWEST checkpoints " + os.path.join(path, newest)+". Please check if it's correct!")
                datas = torch.load(os.path.join(path, newest))

    # process the datas
    return datas
