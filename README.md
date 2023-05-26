# 说明文档

## 运行方式

预处理部分：

1. 进入`data_preprocess`目录的各子目录，执行`*.py`文件，得到`pro_feat.npz`和`pro_skill_sparse.npz`文件

2. `python extract.py --dataset=assist09`，根据`dataset`参数选定数据集，得到`pro_pro_sparse.npz`和`skill_skill_sparse.npz`文件

预训练部分：

CUDA_VISIBLE_DEVICES=0 python pebg.py --dataset=assist09
