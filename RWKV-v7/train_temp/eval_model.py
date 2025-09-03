#!/usr/bin/env python3
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# previous version only support lm_eval==0.3.0
# this version support lm_eval>=0.4.0
#
import os, sys, types, json, math, time
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

# 设置环境变量
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
# 添加CUDA架构环境变量，避免编译问题
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0 7.5 8.0 8.6"

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ["RWKV_MY_TESTING"] = 'x070'
os.environ["RWKV_HEAD_SIZE"] = '64'

# 在导入lm_eval之前，备份并替换get_tokenizer函数
from lm_eval.tasks.ruler.common_utils import get_tokenizer as original_get_tokenizer

def patched_get_tokenizer(tokenizer=None, pretrained=None, **kwargs):
    """修复get_tokenizer函数，处理字典类型的参数"""
    # 如果传入的是字典，则提取其中的字符串值
    if isinstance(tokenizer, dict):
        # 从字典中提取pretrained或tokenizer键的值
        if "pretrained" in tokenizer:
            pretrained = tokenizer["pretrained"]
        elif "tokenizer" in tokenizer:
            pretrained = tokenizer["tokenizer"]
        tokenizer = None
    elif isinstance(pretrained, dict):
        if "pretrained" in pretrained:
            pretrained = pretrained["pretrained"]
        elif "tokenizer" in pretrained:
            pretrained = pretrained["tokenizer"]
    
    # 确保至少有一个有效的参数
    pretrained = tokenizer or pretrained
    if not pretrained:
        pretrained = "gpt2"  # 默认使用gpt2
    
    print(f"Using tokenizer {pretrained} for synthetic tasks.")
    return original_get_tokenizer(pretrained, **kwargs)

# 替换函数
import lm_eval.tasks.ruler.common_utils
lm_eval.tasks.ruler.common_utils.get_tokenizer = patched_get_tokenizer

from lm_eval import tasks, evaluator, utils
from lm_eval.models.huggingface import HFLM
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.group import ConfigurableGroup

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

TASK_TO_NUM_FEWSHOT = {
    'mmlu': 5,
}    

########################################################################################################

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs/lm_eval/', help='Directory to save logs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run evaluation on')
    parser.add_argument('--ctx_len', type=int, default=1024, help='Context length for the model')
    
    # 评估任务相关参数
    tasks_list=['lambada_openai','piqa','hellaswag','winogrande','arc_challenge','arc_easy','headqa','openbookqa','sciq','triviaqa','record','copa','ruler']
    group = parser.add_argument_group('eval')
    group.add_argument('--tasks', type=str, nargs='+', default=[tasks_list[3]], 
                      help='Tasks to evaluate on')

    args = parser.parse_args()
    return args

args = parse_config()
MODEL_NAME = Path(args.model_path)
OUTPUT_DIR = Path(args.log_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading model - {MODEL_NAME}')

# 加载模型和分词器
def load_model_and_tokenizer(model_path, device='cuda'):
    # 添加当前目录到Python路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 导入必要的模块
    from src.model import RWKV
    
    # 加载模型权重
    weights = torch.load(model_path, map_location='cpu')
    
    # 从权重中推断模型参数
    n_layer = 0
    for k in weights.keys():
        if k.startswith("blocks.") and k.endswith(".ln1.weight"):
            layer_id = int(k.split(".")[1])
            n_layer = max(n_layer, layer_id + 1)
    
    # 获取嵌入维度
    n_embd = weights['emb.weight'].shape[1]
    vocab_size = weights['emb.weight'].shape[0]
    
    # 确定head_size和dim_att
    head_size = int(os.environ.get("RWKV_HEAD_SIZE", 64))
    dim_att = n_embd
    
    # 创建参数对象，补充缺失的参数
    class Args:
        def __init__(self):
            self.n_layer = n_layer
            self.n_embd = n_embd
            self.ctx_len = args.ctx_len
            self.vocab_size = vocab_size
            self.head_size = head_size
            self.dim_att = dim_att
            self.dim_ffn = int((n_embd * 3.5) // 32 * 32)
            self.grad_cp = 0
            self.weight_decay = 0.0
            self.lr_init = 1e-5
            self.lr_final = 1e-5
            self.betas = (0.9, 0.99)
            self.adam_eps = 1e-8
            self.accelerator = "gpu" if "cuda" in device else "cpu"
            self.my_testing = 'x070'
            self.train_stage = 0
            self.warmup_steps = 0
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.micro_bsz = 1
            self.num_nodes = 1
            self.devices = 1
            self.precision = "bf16"
            self.strategy = "deepspeed_stage_2"
            self.enable_progress_bar = False
            self.ds_bucket_mb = 200
            self.epoch_count = 1
            self.epoch_begin = 0
            self.epoch_save = 1
            self.review_base_range = 3
    
    model_args = Args()
    
    # 创建模型实例
    model = RWKV(model_args)
    
    # 加载权重
    model.load_state_dict(weights)
    model = model.to(device).bfloat16()  # 使用bfloat16精度
    model.eval()
    
    # 定义并加载分词器
    class TRIE_TOKENIZER():
        def __init__(self, file_name):
            self.idx2token = {}
            sorted = [] # must be already sorted
            with open(file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for l in lines:
                idx = int(l[:l.index(' ')])
                x = eval(l[l.index(' '):l.rindex(' ')])
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes)
                assert len(x) == int(l[l.rindex(' '):])
                sorted += [x]
                self.idx2token[idx] = x

            self.token2idx = {}
            for k, v in self.idx2token.items():
                self.token2idx[v] = int(k)

            # precompute some tables for fast matching
            self.table = [[[] for j in range(256)] for i in range(256)]
            self.good = [set() for i in range(256)]
            self.wlen = [0 for i in range(256)]

            for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
                s = sorted[i]
                if len(s) >= 2:
                    s0 = int(s[0])
                    s1 = int(s[1])
                    self.table[s0][s1] += [s]
                    self.wlen[s0] = max(self.wlen[s0], len(s))
                    self.good[s0].add(s1)

        def encodeBytes(self, src: bytes) -> list[int]:
            src_len: int = len(src)
            tokens: list[int] = []
            i: int = 0
            while i < src_len:
                s: bytes = src[i : i + 1]

                if i < src_len - 1:
                    s1: int = int(src[i + 1])
                    s0: int = int(src[i])
                    if s1 in self.good[s0]:
                        sss: bytes = src[i : i + self.wlen[s0]]
                        try:
                            s = next(filter(sss.startswith, self.table[s0][s1]))
                        except:
                            pass
                tokens.append(self.token2idx[s])
                i += len(s)

            return tokens

        def encode(self, src: str):
            return self.encodeBytes(src.encode("utf-8"))

        def decodeBytes(self, tokens):
            result = b''
            for i in tokens:
                token_bytes = self.idx2token.get(i)
                if token_bytes is not None:
                    result += token_bytes
            try:
                return result.decode('utf-8')
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，使用errors='replace'来替换无法解码的字符
                return result.decode('utf-8', errors='replace')

        def decode(self, tokens):
            return self.decodeBytes(tokens)

        def printTokens(self, tokens):
            for i in tokens:
                s = self.idx2token[i]
                try:
                    s = s.decode('utf-8')
                except:
                    pass
                print(f'{repr(s)}{i}', end=' ')
                # print(repr(s), i)
            print()
    
    # 加载分词器，使用RWKV-v7目录下的词汇表文件
    tokenizer_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rwkv_vocab_v20230424.txt")
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

eval_tasks = args.tasks
print(f"Evaluating on tasks: {eval_tasks}")

# 设置few-shot数量
num_fewshot = {task: TASK_TO_NUM_FEWSHOT.get(task, 0) for task in eval_tasks}

# 定义PAD和STOP token
RWKV_PAD = tokenizer.encode('\n')  # 使用'\n'作为PAD
STOP_TOKEN = RWKV_PAD + tokenizer.encode('\n\n')  # 使用'\n\n'作为STOP
print('RWKV_PAD', RWKV_PAD)
print('STOP_TOKEN', STOP_TOKEN)

########################################################################################################

logitBuf = {}
correctBuf = {}

@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0
        self.bos_token_id = 0
        self.pad_token_id = 0
        
    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)
        
    def __call__(self, string: str):
        input_ids = torch.LongTensor(self.encode(string))
        return TokenizerOutput(input_ids=input_ids)
    
    # 添加必要的属性和方法以满足lm-eval的要求
    @property
    def vocab_size(self):
        return 65536  # 根据实际词汇表大小调整
    
    @property
    def eos_token(self):
        return "</s>"
    
    @property
    def bos_token(self):
        return "<s>"
    
    @property
    def pad_token(self):
        return "<pad>"
        
    def get_vocab(self):
        # 返回空字典，因为RWKV使用字节级词汇表
        return {}
        
    # 使对象可哈希化，解决"unhashable type: 'dict'"错误
    def __hash__(self):
        return hash(id(self))
    
    def __eq__(self, other):
        return id(self) == id(other)
        
    # 添加__repr__方法使对象字符串化
    def __repr__(self):
        return f"TokenizerWrapper(id={id(self)})"
        
    # 添加__str__方法
    def __str__(self):
        return self.__repr__()
        
    # 为兼容transformers tokenizer添加额外方法
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return str(ids)
        return [str(id) for id in ids]
        
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            try:
                return int(tokens)
            except ValueError:
                return 0
        return [self.convert_tokens_to_ids(token) for token in tokens]

class EvalHarnessAdapter(HFLM):
    def __init__(self):
        # 初始化但不设置模型参数
        self._batch_size = 1
        self.tokenizer = TokenizerWrapper(tokenizer)
        # 确保有device属性
        self._device = next(model.parameters()).device if model is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self):
        # 返回self作为model的占位符
        return self
    
    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        return args.ctx_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1
    
    @property
    def max_new_tokens(self):
        return 64

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _model_call(self, inps):
        """
        计算模型在输入tokens上的输出logits
        """
        with torch.no_grad():
            return model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        """
        为ruler等任务提供生成接口
        """
        # 这里实现一个简单的生成方法
        with torch.no_grad():
            # 简化版本的生成，实际使用中可能需要更复杂的实现
            outputs = model(context)
            # 只返回最后一维，模拟生成结果
            return outputs[:, -1:, :]

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        for COUNTER in tqdm(range(len(requests)), " Running loglikelihood requests"):
            n = COUNTER
            raw_src = requests[n][0][0] + requests[n][0][1]

            src = requests[n][1] + requests[n][2]

            raw_src = '\n' + raw_src
            src = RWKV_PAD + src

            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(RWKV_PAD)
                logit = 0
                
                with torch.no_grad():
                    outputs = model.forward(torch.LongTensor([src]).to(model.device))[0]
                    for i in range(q_len-1, len(src)-1):
                        oo = outputs[i].detach().float()
                        dst = src[i+1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            
            res += [(logit, correct)]
        return res
    
    @torch.no_grad()
    def greedy_generate(self, ctx):
        all_tokens = []
        out_last = 0
        out_str = ''
        tokens = self.tokenizer.encode(ctx)
        tokens = torch.LongTensor([tokens]).to(model.device)
        for i in range(self.max_new_tokens):
            out = model.forward(tokens)[:, -1:, :]
            token = out.argmax(dim=-1)
            tokens = torch.cat([tokens, token], dim=1)
            token = token.item()
            if token in STOP_TOKEN:
                break
            all_tokens += [token]
            tmp = self.tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp:  # is valid utf-8 string?
                out_str += tmp
                out_last = i + 1
        return out_str
    
    @torch.no_grad()
    def generate_until(self, requests):
        """
        Generate until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        res = []
        # get only the args from each Instance object
        reqs = [req.args for req in requests]

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(reqs, _collate)
        for context, gen_kwargs in tqdm(reord.get_reordered(), "Running greedy generation"):
            out_str = self.greedy_generate(context)
            for term in gen_kwargs['until']:
                out_str = out_str.split(term)[0]
            res.append(out_str)
            torch.cuda.empty_cache()
        return reord.get_original(res)

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=None, limit=None, bootstrap_iters=0):
        ''' Run evaluation on the tasks, such as MMLU, HellaSwag, LAMBADA, etc.
        :param eval_tasks: list of task names to evaluate on
        :param num_fewshot: number of few-shot examples to evaluate on
        :param bootstrap_iters: Set to 0 for skipping all stderr calculations
        '''
        def recursive_set_config(obj, key, value):
            if isinstance(obj, ConfigurableTask):
                obj.set_config(key=key, value=value)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    recursive_set_config(v, key, value)

        if num_fewshot is None:
            num_fewshot = {}

        task_dict = tasks.get_task_dict(eval_tasks)
        for task_name in task_dict:
            task_obj = task_dict[task_name]
            if isinstance(task_name, str):
                task_fewshot = num_fewshot.get(task_name, 0)
            if isinstance(task_name, ConfigurableGroup):
                group_or_task_name = task_name.group_name
                task_fewshot = num_fewshot.get(group_or_task_name, 0)
            if isinstance(task_obj, tuple):
                _, task_obj = task_obj
                if task_obj is None:
                    continue
            if isinstance(task_obj, ConfigurableTask):
                task_obj.set_config(key="num_fewshot", value=task_fewshot)
                print(f"Task {task_name} is a ConfigurableTask, set num_fewshot to {task_fewshot}")
            if isinstance(task_obj, dict):
                print(f"Task {task_name} is a dict, recursing set it to {task_fewshot}")
                recursive_set_config(task_obj, "num_fewshot", task_fewshot)
        
        results = evaluator.evaluate(
                lm=self,
                task_dict=task_dict,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
            )
        return results

adapter = EvalHarnessAdapter()
print(f'Running evaluation on {eval_tasks} with {num_fewshot}-shot examples')
results = adapter.run_eval(
    eval_tasks=eval_tasks,
    num_fewshot=num_fewshot,
)

# 保存结果
eval_results = results['results']

# 转换结果为表格
import pandas as pd
df = pd.DataFrame(eval_results)
task_str = '-'.join(eval_tasks)
model_stem = Path(MODEL_NAME).stem
metric_output_name = model_stem + "_" + task_str + ".csv"
metric_output_path = OUTPUT_DIR / metric_output_name
df.to_csv(metric_output_path)
print(f"Evaluation results saved to {metric_output_path}")

# 打印结果
print("Evaluation results:")
import pprint
pprint.pprint(eval_results)