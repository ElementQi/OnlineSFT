
<table>
  <tr>
    <td style="width: 45%;"><img src="figures/qwenmath_2pic.png" alt="base_figure" style="width: 100%;"></td>
    <td style="width: 45%;"><img src="figures/average_performance_with_baselines.png" alt="training_figure" style="width: 100%;"></td>
  </tr>
</table>


# Online SFT

Code for [Online SFT for LLM Reasoning](https://arxiv.org/abs/2509.xxxxx).

OSFT is a self-help, reward-free method that improves LLM reasoning by finetuning the model on its own generated responses.

> [!NOTE]
> This codebase is not fully optimized (e.g. unnecessary loggings and computation; not tested on the newest version of verl and transformers); feel free to raise any issues you encountered.

## Setup

```sh
git clone https://github.com/ElementQi/OnlineSFT.git
cd OnlineSFT/osft

conda create -n osft python=3.10
conda activate osft

# some machines need this
# conda install -c conda-forge pyzmq
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation  # choose a suitable version for your own machine
pip install -e . --no-dependencies
```

> [!NOTE]
> Installation process could be different from various machines and system, but the installation script above is tested on our A800 and 3090 clusters.

If you have any issues when installing the packages, you can refer to the [installation tutorial from verl](https://verl.readthedocs.io/en/latest/start/install.html) and check [our runnable conda environmrnt](osft/osft_envs/environment.yaml).

## Reproduce Experiments

Before running the scripts below, please ensure you are in the project root directory (`OnlineSFT/osft`).

> [!WARNING]
> **Configuration is required before running!**
> You MUST modify the scripts inside the `examples/` directory to fit your environment. Key parameters to change include:
> - Model Path: Update the path to your base model.
> - Logger: The default logger is `swanlab`. If you prefer `wandb` or just `console`, modify `trainer.logger=['console','swanlab']`.
> - Shared Memory: When setting `actor_rollout_ref.model.use_shm=True`, ensure the model name is a valid absolute path on your machine.
> - Other Hyperparameters: Like rollout number, batch size...


```sh
conda activate osft

# for OSFT training
bash examples/osft_1e7_dsr_tau0s6.sh

# for GRPO training
bash examples/grpo_1e7_dsr_tau1s.sh

# for DAPO and Dr. GRPO training
bash examples/dapo_1e7_dsr_tau1s.sh
bash examples/drgrpo_1e7_dsr_tau1s.sh
```


### Validation on specific checkpoints

For evaluation, we re-write the evaluation code for `_validate` function inside the Trainer. And if you want to evaluate a specific checkpoint or base models, try to use `recipe/osft/generation_same_validate.py`.

This is an example for sweeping temperatures for a base model:

```
bash examples/all_models_tauv.sh
```

If you want to transform the training checkpoints via verl, you should follow the instructions from [verl official tutorial for model converting](https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model). The model merge script is located in [here](https://github.com/volcengine/verl/blob/main/scripts/legacy_model_merger.py).

## Dataset

The datasets are located in `osft/data` folder. There are two training sets DSR and Openthoughts math-only. And there are 6 benchmark files inside its `benchmarks` folder.

## What we mainly modified

### Training logic

- `recipe/osft/osft_trainer`: SFT training logic

- `recipe/osft/dp_actor`: Cross entropy calculating


### Validation logic

- `verl/trainer/ppo/metric_utils.py`: Added a pass@k calculation logic during validation.

- `verl/trainer/ppo/ray_trainer.py`: Modified the validation logic by selecting 16 different question-response pairs for different validation dataset in `_validate` function.

### Verifier

`osft/verl/utils/reward_score/__init__.py`

## Source Acknowledgement

This repository is built based on [VERL](https://github.com/volcengine/verl) at commit hash `38d9a88170786a45cb189a08290c4651e6d6f671`. 


For verifier, we used the verifier from [The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning.](https://arxiv.org/pdf/2505.22617), which uses [HuggingFace Math-Verify](https://github.com/huggingface/Math-Verify). The source code could be found in [VERL entropy recipe](https://github.com/volcengine/verl/tree/main/recipe/entropy/reward_score/entropy_math).


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{placeholder,
  title   = {Online SFT for LLM Reasoning},
  author  = {placeholder},
  journal = {arXiv preprint arXiv:2509.xxxxx},
  year    = {2025}
}
```