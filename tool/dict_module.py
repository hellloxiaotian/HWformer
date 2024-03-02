from collections import OrderedDict
import torch
import os

dict_path = '/data/zmh/result/models/dwd/color/25.0/model_086_sigma25.pth'

load_dict = torch.load(dict_path)

for key in list(load_dict.keys()):
    load_dict[key[6:]] = load_dict.pop(key)

torch.save(load_dict, dict_path)