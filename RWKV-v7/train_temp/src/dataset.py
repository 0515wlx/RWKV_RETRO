########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
import torch.distributed as dist

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        rank_zero_info("Loading data file...")
        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")
        rank_zero_info(f"Data file loaded successfully. Buffer size: {len(self.data._bin_buffer)}, Dtype size: {self.data._index._dtype_size}")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1
        
        # 初始化分布式训练相关变量
        self.global_rank = 0
        self.world_size = 1
        self.real_epoch = 0
        
        # 检查是否使用了分布式训练
        if dist.is_available() and dist.is_initialized():
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            rank_zero_info(f"Distributed training initialized. Rank: {self.global_rank}, World Size: {self.world_size}")
        else:
            rank_zero_info("Distributed training not initialized.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        # 添加更详细的边界检查
        if i < 0:
            rank_zero_info(f"Warning: Calculated index i={i} is negative. Setting to 0.")
            i = 0
            
        if i + req_len > self.data_size:
            rank_zero_info(f"Warning: Index i={i} is too close to the end of the dataset. Adjusting from {i} to {self.data_size - req_len - 1}")
            i = self.data_size - req_len - 1
            
        try:
            dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)
        except Exception as e:
            rank_zero_info(f"Error loading data at index {i} with length {req_len}: {e}")
            # 返回一个默认的数组以避免程序崩溃
            dix = np.zeros(req_len, dtype=int)
            # 添加更多调试信息
            rank_zero_info(f"Current state - epoch: {epoch}, idx: {idx}, rank: {rank}, world_size: {world_size}, ctx_len: {ctx_len}")

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y
