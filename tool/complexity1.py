'''
How to use:
    no transformer:
        python Complexity.py --model_name $model_name --n_colors 3 --n_GPUs 1 --GPU_id 3
    has transformer:
        should add configuration of transformer：
        python Complexity.py --model_name $model_name --n_colors 3  --patch_size 96 --patch_dim 2 --num_heads 8 --num_layers 5 --n_GPUs 1 --GPU_id 2 --flag 6
To measure：
|——Number of parameters——Param
 ——Computational complexity——GFLOPs
CMD: python complexity1.py --model_name bformer --mode train --n_colors 3  --patch_size 192 --patch_dim 3 --num_heads 9 --num_layers 8 --n_GPUs 1 --GPU_id 2
python complexity1.py --model_name bformer --mode test --n_colors 3  --patch_size 192 --patch_dim 3 --num_heads 9 --num_layers 8 --n_GPUs 1 --GPU_id 2

——runing time：
    3*256*256
    3*512*512
    3*1024*1024
'''

import os
import sys
from importlib import import_module

sys.path.append("..")

import torch
from ptflops import get_model_complexity_info

from main.option import args


os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import model
'''
module = import_module('model.' + args.model_name.lower())
_model, _ = module.make_model(args)
'''
with torch.no_grad():
    # _model = model.Model(args)
    #
    # macs, params = get_model_complexity_info(_model,
    #                                          (3, 192, 192),
    #                                          as_strings=False,
    #                                          print_per_layer_stat=False,
    #                                          verbose=True)
    # print('{:<30}   {:<8}'.format('Computational complexity(GFLOPs):', macs/(2e9)))
    # print('{:<30}   {:<8}'.format('Number of parameters(Param,M):', params/(1e6)))

    import time

    _model = model.Model(args).to(device)

    if args.mode == "train":
        input = torch.randn((1, 3, 96, 96)).type(torch.FloatTensor).to(device)

        torch.cuda.synchronize()
        time_start = time.time()
        #在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('96*96\ttotally cost', time_end-time_start)

        torch.cuda.synchronize()
        time_start = time.time()
        #在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('96*96\ttotally cost', time_end-time_start)

        input = torch.randn((1, 3, 192, 192)).type(torch.FloatTensor).to(device)

        torch.cuda.synchronize()
        time_start = time.time()
        # 在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('192*192\ttotally cost', time_end - time_start)

        torch.cuda.synchronize()
        time_start = time.time()
        # 在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('192*192\ttotally cost', time_end - time_start)

    elif args.mode == "test":
        input = torch.randn((1, 3, 192, 192)).type(torch.FloatTensor).to(device)

        torch.cuda.synchronize()
        time_start = time.time()
        # 在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('192*192\ttotally cost', time_end - time_start)

        torch.cuda.synchronize()
        time_start = time.time()
        # 在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('192*192\ttotally cost', time_end - time_start)
        
        input = torch.randn((1, 3, 256, 256)).type(torch.FloatTensor).to(device)

        torch.cuda.synchronize()
        time_start = time.time()
        #在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('256*256\ttotally cost', time_end-time_start)

        torch.cuda.synchronize()
        time_start = time.time()
        #在这里运行模型
        _model(input)
        torch.cuda.synchronize()
        time_end = time.time()
        print('256*256\ttotally cost', time_end-time_start)
    #
    #
    # input = torch.randn((1, 3, 512, 512)).type(torch.FloatTensor).to(device)
    # torch.cuda.synchronize()
    # time_start = time.time()
    # #在这里运行模型
    # _model(input)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print('512*512\ttotally cost', time_end-time_start)
    #
    # input = torch.randn((1, 3, 1024, 1024)).type(torch.FloatTensor).to(device)
    # torch.cuda.synchronize()
    # time_start = time.time()
    # #在这里运行模型
    # _model(input)
    # torch.cuda.synchronize()
    # time_end = time.time()
    # print('1024*1024\ttotally cost', time_end-time_start)