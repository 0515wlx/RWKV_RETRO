- fork from https://github.com/BlinkDL/RWKV-LM
- eval脚本参考了https://github.com/howard-hou/RWKV-X
- 包含的完整的训练和评估流程
- 模型参数94.9M
- 配置和https://github.com/Alic-Li/Mini_RWKV_7 一样，参数稍微多了一点是因为改了模型架构，多了个review_mlp矩阵
- 训练207个epoch，前205个ctx=512，第206个ctx=1024,第207个ctx=8192
- loss看train_log.txt
- 数据集：https://huggingface.co/datasets/BlinkDL/minipile-tokenized
- 评估结果：
,arc_challenge
alias,arc_challenge
"acc,none",0.18344709897610922
"acc_stderr,none",N/A
"acc_norm,none",0.21331058020477817
"acc_norm_stderr,none",N/A

,arc_easy
alias,arc_easy
"acc,none",0.4015151515151515
"acc_stderr,none",N/A
"acc_norm,none",0.36574074074074076
"acc_norm_stderr,none",N/A

,copa
alias,copa
"acc,none",0.64
"acc_stderr,none",N/A

,headqa_en,headqa_es
alias,headqa_en,headqa_es
"acc,none",0.23668854850474105,0.21845368344274252
"acc_stderr,none",N/A,N/A
"acc_norm,none",0.2786287381473377,0.24653537563822028
"acc_norm_stderr,none",N/A,N/A

,hellaswag
alias,hellaswag
"acc,none",0.27126070503883687
"acc_stderr,none",N/A
"acc_norm,none",0.2847042421828321
"acc_norm_stderr,none",N/A

,lambada_openai
alias,lambada_openai
"perplexity,none",96.69273036346527
"perplexity_stderr,none",N/A
"acc,none",0.24762274403260237
"acc_stderr,none",N/A

,niah_multikey_1
alias,niah_multikey_1
"4096,none",0.136
"4096_stderr,none",N/A

,niah_multikey_2
alias,niah_multikey_2
"4096,none",0.0
"4096_stderr,none",N/A

,niah_multikey_3
alias,niah_multikey_3
"4096,none",0.0
"4096_stderr,none",N/A

,niah_multiquery
alias,niah_multiquery
"4096,none",0.116
"4096_stderr,none",N/A

,niah_multivalue
alias,niah_multivalue
"4096,none",0.0745
"4096_stderr,none",N/A

,niah_single_1
alias,niah_single_1
"4096,none",0.002
"4096_stderr,none",N/A

,niah_single_2
alias,niah_single_2
"4096,none",0.058
"4096_stderr,none",N/A

,niah_single_3
alias,niah_single_3
"4096,none",0.008
"4096_stderr,none",N/A

,openbookqa
alias,openbookqa
"acc,none",0.146
"acc_stderr,none",N/A
"acc_norm,none",0.262
"acc_norm_stderr,none",N/A

,piqa
alias,piqa
"acc,none",0.5952121871599565
"acc_stderr,none",N/A
"acc_norm,none",0.6001088139281828
"acc_norm_stderr,none",N/A

,ruler_cwe
alias,ruler_cwe
"4096,none",0.0
"4096_stderr,none",N/A

,ruler_fwe
alias,ruler_fwe
"4096,none",0.015999999999999993
"4096_stderr,none",N/A

,ruler_qa_hotpot
alias,ruler_qa_hotpot
"4096,none",0.058
"4096_stderr,none",N/A

,ruler_qa_squad
alias,ruler_qa_squad
"4096,none",0.023333333333333327
"4096_stderr,none",N/A

,ruler_vt
alias,ruler_vt
"4096,none",0.0683999999999999
"4096_stderr,none",N/A

,sciq
alias,sciq
"acc,none",0.695
"acc_stderr,none",N/A
"acc_norm,none",0.602
"acc_norm_stderr,none",N/A

,triviaqa
alias,triviaqa
"exact_match,remove_whitespace",0.0011145786892554615
"exact_match_stderr,remove_whitespace",N/A

,winogrande
alias,winogrande
"acc,none",0.5082872928176796
"acc_stderr,none",N/A
