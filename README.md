# SLM-1B Challenge

An open research project exploring how much a language model with approximately one billion parameters can achieve through better data, efficient computation, reasoning, and tools.

The central question is simple: how far can we push a compact model while keeping its development reproducible, its behavior observable, and its artifacts openly reusable?

The project aims to develop a useful model for local deployment and to understand which techniques improve its capabilities under realistic resource constraints. Success means measurable progress in task completion, reliability, and efficiency, supported by experiments that others can inspect and reproduce.

## Project status

SLM-1B Challenge is currently at the research planning stage. This repository starts with the project vision and experimental agenda. It does not yet provide a model, training pipeline, dataset release, or benchmark results. The methods described below are proposed directions whose value at this scale remains to be tested.

## Open science as a design requirement

The ambition is to release the source code, model weights, tokenizer, configurations, training recipes, and evaluation procedures. Dataset provenance, preparation scripts, experiment logs, and both positive and negative findings should make the development process inspectable.

Openness must be documented for each component. A public download alone does not establish that an artifact meets the project's reuse requirements. The intended core corpus will require identifiable sources, recorded permissions, and a review of applicable obligations before inclusion.

Apache 2.0 is a candidate license for original code and model artifacts where the necessary rights are established. This README does not grant that license. Formal license files and release documentation will define the terms of each published artifact, while third party materials retain their applicable terms.

## Research directions

### Reasoning and adaptive computation

We plan to compare explicit reasoning, latent reasoning, and combinations of the two. Coconut and HybridAgent are starting points for this investigation. Recurrent depth and adaptive computation are additional directions for studying whether a compact model can benefit from spending more computation on difficult tasks.

Experiments will measure answer quality, task completion, latency, memory use, and total inference cost. Reducing visible tokens is useful only when the resulting system remains effective and its total cost supports the claimed improvement.

### Attention and information flow

This track will examine how attention mechanisms affect information flow, representation similarity, and performance as context length increases. Candidate experiments include comparisons between standard dot product attention, gated alternatives, and configurations that vary across layers.

Attention sinks are a phenomenon to investigate rather than an assumed defect. Findings from graph models motivate questions, but do not establish benefits for an autoregressive language model. Initial architecture experiments are planned at roughly 100 million to 300 million parameters before considering a transfer to the one billion parameter target.

### Tools, memory, and adaptive agent environments

The agent environment includes instructions, tools, memory, skills, scripts, plugins, execution rules, and optional specialized agents. We plan to compare the model alone, the model with a fixed environment, and the model with an environment that can propose and evaluate changes to its configuration.

Harness² motivates this research direction. A central question is whether a compact model can improve its own environment reliably and economically. Any adaptation must be evaluated for transfer to unseen tasks, regressions, and the computation consumed while searching for improvements.

Agent experiments will begin in controlled environments with explicit permissions, recorded actions, and verifiable outcomes.

### Causal explainability and observability

Readable explanations and attention visualizations do not, by themselves, establish why a model made a decision. We plan to test explanatory claims through interventions such as removing, replacing, or perturbing selected internal states and measuring changes in behavior.

Agent traces should connect task inputs, relevant execution context, tool calls, observable decisions, verification results, and resource costs. Experiments will assess whether those traces help locate failures and support recovery when tools return incorrect information, time out, or introduce conflicting instructions.

## Open Data Factory

The data pipeline is intended to be a reproducible research artifact in its own right. It will track source revisions, licenses, attribution requirements, transformations, filtering decisions, and dataset splits.

Common Pile and selected SmolTalk2 subsets are initial candidates for review. FineWeb, FineWeb2, Dolma, The Stack v2, and Tülu mixtures are additional candidates for comparison or selective use. These names form a research shortlist, not an approved training corpus. Inclusion will depend on review of the specific sources and subsets used.

Preparation will include quality filtering, exact and approximate deduplication, privacy review, and checks for benchmark contamination. The intended core corpus excludes sources with unidentified provenance or restrictions incompatible with the project's reuse goals, including NonCommercial and NoDerivatives restrictions.

We also plan to create synthetic tasks with verifiable outcomes, including code exercises with tests, simulated API calls, document retrieval, and operations on fictional files. Their records should preserve the generator, any teacher model, the generation conditions, and the verification procedure.

Data mixtures and curricula will be experimental variables. General text, science, mathematics, code, French, and English are areas of interest; instruction and agent trajectories will be considered separately for post training. No mixture is currently established as the training recipe.

## Experimental approach

Two development paths will support the research. Experiments with an existing checkpoint can test reasoning and agent methods sooner, subject to review of its terms and provenance. Training from scratch offers greater control over architecture, data, and the resulting scientific claims. Results from these paths will be reported separately.

Each experiment should define a baseline, a hypothesis, a controlled change, and a reproducible evaluation procedure. Comparisons will account for training data, parameter count, available tools, and computation budgets. Where practical, repeated runs will help distinguish consistent gains from variation.

Evaluation will measure accuracy, verified task completion, robustness, latency, peak memory, tokens, tool calls, retries, and energy where it can be measured. Cost per successfully completed task will include failed attempts and environment adaptation overhead. Training and preparation costs will also be reported so that cheaper inference is not mistaken for lower total cost.

Memory experiments will vary context length, introduce contradictory information, and test retrieval failures. Agent experiments will include tool errors and unfamiliar tasks. Final evaluation data must remain separate from training and from the tasks used to optimize the agent environment.

## Planned progression

The first milestone is a reproducible baseline with a documented data selection process and evaluation suite. Small architecture experiments and controlled agent tasks will then establish which directions justify larger runs.

Promising methods will be tested individually before being combined at the target scale. A release should include the artifacts and documentation needed to reproduce its reported results, together with known limitations and unsuccessful experiments.

## Contributing

Useful contributions include reproducible baselines, dataset provenance reviews, controlled ablations, evaluation environments, and tools for inspecting failures. Research proposals should explain the question, the comparison, the expected resource requirements, and the evidence that would support or reject the hypothesis.

The project's guiding question remains open: what capabilities can a transparent, reproducible model of one billion parameters achieve when its data, computation, and tools are developed and evaluated together?
