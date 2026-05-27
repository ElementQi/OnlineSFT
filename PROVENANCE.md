# Provenance

This repository contains the SePT implementation for the paper:

> A Model Can Help Itself: Reward-Free Self-Training for LLM Reasoning
> arXiv:2510.18814

## Project Origin

SePT stands for Self-Evolving Post-Training. Earlier development versions of
this project used the names OnlineSFT and OSFT. The repository was later renamed
to SePT to match the paper terminology.

The implementation is built on top of VERL:

- Upstream project: https://github.com/verl-project/verl
- Upstream base commit: `38d9a88170786a45cb189a08290c4651e6d6f671`
- Upstream base commit URL:
  https://github.com/verl-project/verl/commit/38d9a88170786a45cb189a08290c4651e6d6f671
- Upstream license: Apache License 2.0

## SePT-Specific Components

The main SePT-specific changes include:

- `sept/recipe/sept/sept_trainer.py`
- `sept/recipe/sept/dp_actor.py`
- `sept/recipe/sept/fsdp_workers.py`
- `sept/recipe/sept/main_sept.py`
- `sept/recipe/sept/config/`
- SePT example scripts under `sept/examples/`
- pass@k and validation changes under `sept/verl/trainer/ppo/`
- verifier integration under `sept/verl/utils/reward_score/`
- SePT datasets and benchmark files under `sept/data/` and
  `sept/data_user_reasoning/`

File-level headers and git history are the source of truth for exact
modification history.

## Licensing

Unless otherwise noted:

- Source code is licensed under the Apache License 2.0. See `LICENSE`.
- Paper, documentation, and repository figures authored by the SePT authors are
  licensed under CC BY 4.0. See `LICENSE-DOCS`.
- Third-party components remain under their own licenses and notices. See
  `NOTICE` and `THIRD_PARTY_NOTICES.md`.

## Attribution Request

The Apache License 2.0 requires preservation of applicable copyright, license,
and NOTICE information when redistributing this work or derivative works. The
project also provides `CITATION.cff` so software and papers that build on SePT
can cite the associated research paper accurately.
