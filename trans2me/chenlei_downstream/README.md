# Protein Representation Learning

## Requirements

- python 3.8
- pytorch 1.9.0
- apex 0.1
- pytorch_lightning 1.4.2
- CUDA 11.1
- numpy 1.19.5
- matplotlib 3.4.2
- bio 0.5.0
- biotite 0.29.0
- lmdb 1.2.1

## Usage

### Protein Pre-training

**Directory: cd pretraining/**

**ESM-1b Pre-training**

```python
python pretrain_uniref50_scratch.py

(Multi-GPU) CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 pretrain_uniref50_scratch_multigpu.py 
```

**MSA Transformer Pre-training**

```python
python pretrain_msa_scratch.py
```

**Test**

```python
python test_attn_esm1b.py

python test_mutation_esm1b.py
```

### Protein Fine-tuning for TAPE Downstream Tasks

**Directory: cd downstream_tape/**

**ESM-1b Fine-tuning**

```python
(Flu) python finetune_task_flu_ori.py

(Sta) python finetune_task_sta_ori.py

(RH) python finetune_task_rh_ori.py

(SSP) python finetune_task_ssp_ori.py

(CP) python finetune_task_cp_ori.py
```

### GREMLIN & Single-layer Attention

**Directory: cd gremlin/**

```python
(Model Choices) python train.py
```

### Knowledge Distillation

**Directory: cd kd/**

**ESM-1b Knowledge Distillation**

```python
python pretrain_kd_esm1b.py
```

**MSA Transformer Knowledge Distillation**

```python
python pretrain_kd_msa.py
```

### MSA ID Embedding

**Directory: cd msa_id/**

**ESM-1b Pre-training**

```python
python pretrain_esm_ssd.py

python pretrain_esm_ssd_id.py
```

**ESM-1b Fine-tuning**

```python
python finetune_task_ssp_ori.py

python finetune_task_ssp_id.py

python finetune_task_cp_ori.py

python finetune_task_cp_id.py
```

### Contrastive Learning

**Directory: cd cl/**

```python
python pretrain_uniref50_cl.py
```

### Protein-adaptive Fine-tuning

**Directory: cd paf/**

```python
python finetune_task_ssp_paf.py

python finetune_task_ssp_paf_soft.py

python finetune_task_cp_paf.py

python finetune_task_cp_paf_soft.py
```




