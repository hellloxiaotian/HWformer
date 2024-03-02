import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.utils.model_zoo
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args, model=None):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.patch_size = args.patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_GPUs = args.n_GPUs
        self.mode = args.mode
        self.save_models = args.save_models
        self.window_size = args.patch_size

        self.g_patch_size = 192
        if model is None or isinstance(model, str): # 训练时传进来是None/测试时有时候传进来是str（目录）
            module = import_module('model.' + args.model_name.lower())
            self.model, ours = module.make_model(args)
            print("If ours: ", ours)
            if ours == 0:
                model = 0
        else: # 如果model不为空，即直接传进来一个可以用的模型
            self.model = model
            print("Model is Created!")

        if self.mode == 'train':
            self.model.train()

            if self.args.pretrain != '':
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(self.args.dir_model, 'pre_train', self.args.model_name, self.args.pre_train)),
                    strict=False)

            # self.model.to(self.device)
            self.model = nn.DataParallel(self.model.to(self.device), device_ids=[i for i in range(self.n_GPUs)])
        elif self.mode == 'test':

            if isinstance(model, str):  # 如果传进来model参数为字符串，表示需要从磁盘加载模型文件
                dict_path = model
                print("Be ready to load model from {}".format(dict_path))

                load_dict = torch.load(dict_path)

                # self.model.load_state_dict(load_dict, strict=True)

                try:
                    self.model.load_state_dict(load_dict, strict=True)
                except RuntimeError:
                    from collections import OrderedDict
                    new_dict = OrderedDict()

                    for key, _ in load_dict.items():    # 去掉开头module.前缀
                        new_dict[key[7:]] = load_dict[key]
                    print(1)
                    self.model.load_state_dict(new_dict, strict=True)

                self.model = nn.DataParallel(self.model.to(self.device), device_ids=[i for i in range(self.n_GPUs)])
            print(next(self.model.parameters()).device)
            self.model.eval()

    def forward(self, x, sigma=None):
        if self.mode == 'train':
            return self.model(x)
        elif self.mode == 'test':
            if self.args.model_name == "dualscformer" or self.args.model_name == "dualscformer1":
                # return self.forward_new1(x)
                return self.forward_chop(x)

            if self.args.num_layers == 0:  # transformer的层数为0，没有使用transformer模块
                if sigma is None:
                    return self.model(x)
                else: # ffdnet
                    return self.model(x, sigma)
            else:
                # return self.model(x)
                return self.forward_chop(x)
                # print(x.shape)
                
                
                # return self.forward_new1(x)
        else:
            raise ValueError("Choose the train or test model......")

    def forward_chop(self, x, shave=12):
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size / 2)

        # print(batchsize,h,w,padsize,shave)

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        # x_unfold = F.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()
        x_unfold = F.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]
        y_h_cut = self.cut_h(x_h_cut, w, w_cut, padsize, shave, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, h_cut, padsize, shave, batchsize)

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(x_h_top, w, w_cut, padsize, shave, batchsize)
        y_w_top = self.cut_w(x_w_top, h, h_cut, padsize, shave, batchsize)

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)

        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append(
                # P.data_parallel(self.model, x_unfold[i * batchsize:(i + 1) * batchsize, ...], range(self.n_GPUs)).cpu()
                self.model(x_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu()
            )

        y_unfold = torch.cat(y_unfold, dim=0)

        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                     ((h - h_cut), (w - w_cut)), padsize,
                                     stride=int(shave / 2))

        y[..., :padsize, :] = y_h_top
        y[..., :, :padsize] = y_w_top

        y_unfold = y_unfold[..., int(shave / 2):padsize - int(shave / 2), int(shave / 2):padsize - int(shave / 2)].contiguous()
        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                           ((h - h_cut - shave), (w - w_cut - shave)),
                                           padsize - shave, stride=int(shave / 2))

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, padsize - shave, stride=int(shave / 2)),
            ((h - h_cut - shave), (w - w_cut - shave)), padsize - shave,
            stride=int(shave / 2))

        y_inter = y_inter / divisor

        y[..., int(shave / 2 ):(h - h_cut) - int(shave / 2),
        int(shave / 2):(w - w_cut) - int(shave / 2)] = y_inter

        y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2), :],
                       y_h_cut[..., int((padsize - h_cut) / 2 + 0.5):, :]], dim=2)
        y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2), :],
                             y_hw_cut[..., int((padsize - h_cut) / 2 + 0.5):, :]], dim=2)
        y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2)],
                       y_w_cat[..., :, int((padsize - w_cut) / 2 + 0.5):]], dim=3)
        return y.cuda()

    def cut_h(self, x_h_cut, w, w_cut, padsize, shave, batchsize):

        x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(
                #P.data_parallel(self.model, x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...], range(self.n_GPUs)).cpu()
                self.model(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu()
            )
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize, (w - w_cut)), padsize, stride=int(shave / 2))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 ):padsize - int(shave / 2 )].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize, (w - w_cut - shave)), (padsize , padsize - shave),
            stride=int(shave / 2))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize, padsize - shave),
                                       stride=int(shave / 2)), (padsize, (w - w_cut - shave)),
            (padsize, padsize - shave), stride=int(shave / 2))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2):(w - w_cut) - int(shave / 2)] = y_h_cut_inter

        return y_h_cut

    def cut_w(self, x_w_cut, h, h_cut, padsize, shave, batchsize):

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            #y_w_cut_unfold.append(P.data_parallel(self.model, x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...], range(self.n_GPUs)).cpu())
            y_w_cut_unfold.append(self.model(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut), padsize), padsize, stride=int(shave / 2))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2):padsize - int(shave / 2), :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave), padsize), (padsize - shave, padsize),
            stride=int(shave / 2))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize - shave, padsize), stride=int(shave / 2)), ((h - h_cut - shave), padsize),
            (padsize - shave, padsize), stride=int(shave / 2))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2):(h - h_cut) - int(shave / 2), :] = y_w_cut_inter

        return y_w_cut

    def forward_new(self, x):
        B, C, H, W = x.shape

        all_h_s = self.g_patch_size - H % self.g_patch_size

        all_w_s = self.g_patch_size - W % self.g_patch_size
        h_s = []
        w_s = []

        if H % self.g_patch_size != 0:
            n_h_s = H // self.g_patch_size
            s_h = all_h_s // n_h_s
            for i in range(n_h_s):
                h_s.append(s_h * i)
            h_s.append(all_h_s)

        if W % self.g_patch_size != 0:
            n_w_s = W // self.g_patch_size
            s_w = all_w_s // n_w_s
            for i in range(n_w_s):
                w_s.append(s_w * i)
            w_s.append(all_w_s)

        x = self.img_partition(x, self.g_patch_size, [h_s, w_s])

        x = self.model(x)

        return self.img_reserve(x, self.g_patch_size, [h_s, w_s], (B, C, H, W))

    def img_partition(self, x, pat_size, stride):
        result = []
        for i, s_x in enumerate(stride[0]):
            for j, s_y in enumerate(stride[1]):
                result.append(x[..., i * pat_size - s_x: i * pat_size - s_x + pat_size,
                              j * pat_size - s_y: j * pat_size - s_y + pat_size])

        x = torch.stack(result).squeeze(1)

        return x

    def img_reserve(self, x, pat_size, stride, img_size):
        B, C, H, W = img_size

        result = torch.ones((B, C, H, W))
        for i, s_x in enumerate(stride[0]):
            for j, s_y in enumerate(stride[1]):
                result[..., i * pat_size - s_x: i * pat_size - s_x + pat_size,
                j * pat_size - s_y: j * pat_size - s_y + pat_size] = x[i * len(stride[0]) + j, ...]

        return result

    def forward_new1(self, x):
        B, C, H, W = x.shape

        result = torch.ones(B, C, H, W)

        n_H = H // self.window_size
        n_W = W // self.window_size

        body = x[:, :, :n_H*self.window_size, :n_W*self.window_size]
        h_tail = x[:, :, :n_H*self.window_size, -self.window_size:]
        w_tail = x[:, :, -self.window_size:, :n_W*self.window_size]
        hw_tail = x[:, :, -self.window_size:, -self.window_size:]

        B_b, C_b, H_b, W_b = body.shape
        B_h, C_h, H_h, W_h = h_tail.shape
        B_w, C_w, H_w, W_w = w_tail.shape

        # body = self.window_partition2batch(body, window_size=self.patch_size)
        # h_tail = self.window_partition2batch(h_tail, window_size=self.patch_size)
        # w_tail = self.window_partition2batch(w_tail, window_size=self.patch_size)

        body = self.model(body)
        h_tail = self.model(h_tail)
        w_tail = self.model(w_tail)
        hw_tail = self.model(hw_tail)

        # body = self.window_reverse(body, (B_b, C_b, H_b, W_b), window_size=self.patch_size)
        # h_tail = self.window_reverse(h_tail, (B_h, C_h, H_h, W_h), window_size=self.patch_size)
        # w_tail = self.window_reverse(w_tail, (B_w, C_w, H_w, W_w), window_size=self.patch_size)

        result[:, :, :n_H*self.window_size, :n_W*self.window_size] = body
        result[:, :, :n_H*self.window_size, -self.window_size:] = h_tail
        result[:, :, -self.window_size:, :n_W*self.window_size] = w_tail
        result[:, :, -self.window_size:, -self.window_size:] = hw_tail

        return result


    def window_partition2batch(self, x, window_size = 96):
        B, C, H, W = x.shape

        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, -1, window_size, window_size).permute(0, 2, 1, 3,
                                                                                                      4).contiguous().view(
            -1, C, window_size, window_size)
        return x

    def window_reverse(self, x, img_size, window_size = 96,):
        B, C, W, H = img_size
        x = x.view(B, -1, C, window_size, window_size).permute(0, 2, 1, 3, 4).contiguous().view(B, C,
                                                                                                H // window_size,
                                                                                                W // window_size,
                                                                                                window_size,
                                                                                                window_size).permute(
            0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, H, W)

        return x
