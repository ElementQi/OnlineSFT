# Third-Party Notices

This file summarizes major third-party components used or adapted in SePT. It
is not exhaustive; file-level headers and upstream licenses remain authoritative.

## VERL

- Project: VERL: Volcano Engine Reinforcement Learning for LLM
- URL: https://github.com/verl-project/verl
- Base commit used by SePT: `38d9a88170786a45cb189a08290c4651e6d6f671`
- Base commit URL:
  https://github.com/verl-project/verl/commit/38d9a88170786a45cb189a08290c4651e6d6f671
- License: Apache License 2.0
- Copyright notice in upstream files:
  `Copyright 2024 Bytedance Ltd. and/or its affiliates`

SePT builds on VERL and modifies training, actor update, validation, and verifier
integration logic for reward-free self-training.

## HuggingFace Math-Verify

- Project: Math-Verify
- URL: https://github.com/huggingface/Math-Verify

SePT uses Math-Verify-related verifier functionality as described in README.md.
If using Math-Verify directly, install and comply with the license and terms of
the upstream project.

## VERL Entropy Recipe / Entropy Mechanism Verifier

- VERL entropy recipe:
  https://github.com/verl-project/verl-recipe/tree/e7f889574b8301cc0f0fc1d57c6d67f31ffeb689/entropy
- Related paper:
  "The Entropy Mechanism of Reinforcement Learning for Large Language Model
  Reasoning"
  https://arxiv.org/pdf/2505.22617

SePT uses verifier-related code and ideas from this line of work as documented
in README.md.

## Datasets and Model Weights

Datasets, model weights, and external services used with SePT may have their own
licenses, terms, access restrictions, or citation requirements. Users are
responsible for checking those terms before redistributing or using them.
