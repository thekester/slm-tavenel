# Roadmap

Milestones follow the progression in the README: a reproducible baseline first, then small controlled experiments, then the 1B run. Each milestone has an exit criterion that can be checked.

## M0. Reproducible infrastructure (done)

Model with hook points, intervention and analysis tools, provenance-tracked data pipeline with license policy, tokenizer training, token shards, deterministic and resumable training, evaluation of bits per byte, verified tasks and cost, Hugging Face export, agent trace schema.

*Exit criterion, met:* the full chain runs on synthetic data, resuming from a checkpoint reproduces an uninterrupted run exactly, and exported weights give the same logits in `transformers`.

## M1. Approved data for a first corpus

- Review Common Pile v0.1 subsets: pin the revision, list component licenses, record obligations, complete the privacy review.
- Decide on the code source: Common Pile code subsets alone, or a permissive-only extraction of The Stack v2.
- Identify openly licensed French sources, since the candidates are mostly English.
- Assemble the evaluation suite and add every evaluation set to the decontamination step.
- Process a first sample of about 10B tokens and publish its manifests and data card.

*Exit criterion:* `slm sources audit` shows the core sources approved, and a 10B-token corpus is rebuilt bit-identically from its manifest by a second machine.

## M2. Tokenizer and 135M ablations

- Train the tokenizer on a declared sample of the approved corpus; measure fertility for English, French and code (T1).
- Run the 135M baseline and arms A1 to A4 with two seeds each, following the experiment protocol.
- Publish the results, including the rejected arms.

*Exit criterion:* a frozen architecture and tokenizer, with each choice backed by an experiment file.

## M3. Transfer at 300M and hyperparameters

- Confirm the adopted changes at 300M.
- Sweep learning rate and batch size at 135M and 300M to set the 1B values.
- Add fully sharded data parallelism and measure MFU on the target hardware.

*Exit criterion:* measured cost per token and stable training at 300M with the planned 1B settings.

## M4. 1B pretraining

- Run the stable phase on the approved corpus and branch decay phases for mixture experiments (D1).
- Release intermediate checkpoints with their manifests and observability logs.

*Exit criterion:* a base model whose weights, data manifests, configurations and logs are public, with results on the evaluation suite.

## M5. Post training and reasoning

- Supervised fine-tuning on reviewed instruction data, then reinforcement learning with verifiable rewards on synthetic tasks.
- Compare explicit, latent (Coconut-style) and recurrent-depth reasoning at matched inference cost.

*Exit criterion:* a released instruct model and a comparison of reasoning methods reported with total cost per solved task.

## M6. Tools and adaptive environments

- Controlled agent environments with explicit permissions and recorded traces (`slm.trace`).
- Model alone, fixed environment, and self-adapting environment, evaluated on held-out tasks including tool failures.

*Exit criterion:* results on unseen tasks including the computation spent on adapting the environment.
