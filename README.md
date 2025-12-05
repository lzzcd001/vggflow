# VGG-Flow

(Preview version only; code cleaning in progress)


<div align="center">
  <img src="assets/teaser.png" width="900"/>
</div>

## Introduction

This is the official implementation of [VGG-Flow](https://arxiv.org/abs/2512.05116) (NeurIPS 2025).

VGG-Flow is an efficient and robust RL finetuning method for flow matching models.


## Getting Started

### Requirements

- Python >= 3.8
- CUDA 12.4
- PyTorch
- TorchVision

With the above installed, run

```
pip install -r requirements.txt
```

### Pretrained Models

We use StableDiffusion-3.

## Finetuning

### Setting Paths

Modify `config/default_config.py` and replace all `"PLACEHOLDER"` strings to the desired values, including logging path, checkpoing saving path and wandb-related values (if you are using Weights & Bias for logging).

### Getting started

Suppose that you aim to finetune StableDiffusion3 on a 2-GPU A100 node. Run `bash run.sh` to finetune SD3 with the reward model of Aesthetic Score.



### Parameters to tune

For different reward models, you may tune the following factors:

- Reward Scale (`config.model.reward_scale`): How biased the finetuned model is towards the reward model. Higher values lead to faster reward convergence and higher reward at convergence, but at the cost of worse prior preservation and worse sample diversity.
- Subsampling Rate (`config.model.timestep_fraction`, between 0 and 1): How many transitions to use in a trajectory.
- Number of Inference Steps (`config.sampling.num_steps`): How many steps to sample images.

## Citation
If you find our work useful to your research, please consider citing:

```
@inproceedings{liu2025vggflow,
  title={Value Gradient Guidance for Flow Matching Alignment},
  author={Liu, Zhen and Xiao, Tim Z. and Liu, Weiyang and Domingo-Enrich, Carles and Zhang, Dinghuai},
  booktitle={NeurIPS},
  year={2025},
}
```


