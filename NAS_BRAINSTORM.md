# NAS Brainstorm for Wildlife Behavior VLM Project

*Created: 2026-05-13.*

This note sketches practical ways to incorporate neural architecture search
(NAS) into the wildlife behavior recognition paper without turning the project
into a full architecture-design paper. The most defensible direction is not to
search an entire new VLM architecture, but to search the lightweight adaptation
and clip-representation choices around existing VLMs.

## Core framing

Treat wildlife behavior recognition as a **resource-aware VLM adaptation search
problem**. The search space can include backbone choice, LoRA adapter topology,
temporal frame representation, and class-collapse mitigation.

This can be positioned as:

> We formulate parameter-efficient wildlife behavior adaptation as a compact
> neural pipeline search problem over backbone, temporal input design, adapter
> architecture, and imbalance-aware loss. The objective is multi-objective:
> maximize accuracy and macro-F1 while minimizing trainable parameters,
> inference latency, and memory cost.

The important point is to keep the scope compact and honest. Unless a formal
NAS algorithm is implemented, call this **NAS-style adapter search**,
**neural pipeline search**, or **resource-aware adapter architecture search**.

## Option 1: Adapter architecture search

Search over the structure of the LoRA adaptation rather than the base model.
This is the cleanest fit with the existing project.

Possible search dimensions:

| Dimension | Candidate values |
|---|---|
| LoRA rank | 4, 8, 16, 32 |
| LoRA target modules | attention only, MLP only, attention + MLP |
| Layer coverage | all layers, top-k layers, alternating layers |
| LoRA dropout | 0.0, 0.05, 0.1 |
| Loss | standard CE, inv-sqrt CE, balanced CE, focal |
| Sampler | uniform, class-balanced |
| Frames per clip | 4, 8, 12, 16 |
| Prompt format | plain label, structured output |

Possible claim:

> A small NAS-style search over adapter topology and class-aware losses finds
> compact LoRA configurations that preserve accuracy while reducing trainable
> parameters and improving minority-class macro-F1.

Why this fits:

- The existing code already supports LoRA training.
- Phase 2 already supports class-aware loss variants.
- The result can be presented as an extension of the existing A/B contribution
  structure rather than a new unrelated paper.

## Option 2: Temporal input architecture search

For VLMs, the temporal architecture is partly how a video clip is represented as
multiple images. Searching this representation is meaningful for behavior
recognition because behavior often depends on motion cues.

Possible search dimensions:

| Dimension | Candidate values |
|---|---|
| Number of frames | 4, 8, 12, 16 |
| Sampling strategy | uniform, center-biased, motion-aware |
| Clip span | full clip, middle segment, fixed seconds |
| Frame ordering | chronological, shuffled control |
| Prompt layout | direct multi-image prompt, grid summary, per-frame captions |

Possible claim:

> Searching the temporal input design improves cross-dataset transfer more
> reliably than changing prompt format, suggesting that clip representation is a
> first-order design variable for VLM behavior recognition.

Why this fits:

- The project already found that 8-frame transfer beats 4-frame transfer.
- This directly connects to wildlife behavior, where temporal evidence matters.
- It can be evaluated with a small number of runs.

## Option 3: Search class-collapse mitigation

Use NAS-style search to turn the class-collapse failure mode into a searched
training recipe.

Existing variants:

- B1: baseline CE
- B2: inv-sqrt class-weighted CE
- B3: focal loss
- B4: class-balanced sampler

Possible extensions:

| Dimension | Candidate values |
|---|---|
| Weight exponent alpha | 0.0, 0.25, 0.5, 0.75, 1.0 |
| Focal gamma | 1.0, 2.0, 3.0 |
| Label smoothing | 0.0, 0.05, 0.1 |
| Sampler strength | none, mild, full class-balanced |
| LoRA rank | 8, 16 |

Here, class weight could be:

```text
weight(class) = count(class)^(-alpha)
```

where `alpha = 0.5` corresponds roughly to the current inv-sqrt weighting.

Possible claim:

> A compact search over imbalance-aware training recipes identifies inv-sqrt
> reweighting as the best accuracy/macro-F1 tradeoff under severe class
> imbalance, converting class collapse from an observed failure mode into a
> targeted method.

Why this fits:

- It builds directly on Contribution B.
- It is cheap compared with full model search.
- It gives reviewers a more systematic reason to trust the chosen B2 method.

## Option 4: Multi-objective NAS / Pareto frontier

This may be the most publishable NAS framing because wildlife deployment has
real compute constraints.

Search objective:

- Maximize validation accuracy.
- Maximize macro-F1.
- Minimize trainable parameters.
- Minimize inference latency.
- Minimize GPU memory.
- Optionally minimize number of frames.

Example result table:

| Candidate | Backbone | Frames | LoRA rank | Loss | Acc | Macro-F1 | Latency |
|---|---|---:|---:|---|---:|---:|---:|
| A | Qwen2.5-VL-3B | 8 | 16 | CE | TBD | TBD | TBD |
| B | Qwen2.5-VL-3B | 8 | 16 | inv-sqrt | TBD | TBD | TBD |
| C | LLaVA-OV-0.5B | 8 | 16 | CE | TBD | TBD | TBD |
| D | Qwen2.5-VL-3B | 4 | 8 | inv-sqrt | TBD | TBD | TBD |

Possible claim:

> Resource-aware adapter search reveals Pareto-efficient VLM configurations
> that approach larger backbones while requiring fewer trainable parameters or
> fewer input frames.

Why this fits:

- The project already has multiple model sizes and latency artifacts.
- It gives a practical reason for NAS beyond novelty.
- It could support a deployment-oriented figure.

## Option 5: Neural pipeline search over model families

This is broader and easier experimentally, but reviewers may view it as AutoML
or hyperparameter search rather than classic NAS.

Search space:

| Dimension | Candidate values |
|---|---|
| Backbone | Qwen3B, Qwen7B, LLaVA0.5B, LLaVA7B, Gemma4B, MobileVLM |
| Adaptation | zero-shot, LoRA, class-aware LoRA, joint LoRA |
| Frames | 4, 8 |
| Prompt style | plain label, structured answer |
| Dataset recipe | in-domain, transfer, joint |

Possible claim:

> Wildlife behavior recognition benefits from neural pipeline search over
> backbone, adaptation, temporal input, and imbalance correction; the best
> searched recipe is not simply the largest model.

Why this fits:

- Many of these results already exist.
- It can organize existing experiments into a coherent search perspective.

Risk:

- This may read as benchmarking rather than NAS unless the search procedure and
  objective are clearly specified.

## Option 6: NAS for specialist video baselines

This is the most traditional NAS angle, but probably the least aligned with the
current VLM-centered paper.

Possible search dimensions for X3D/SlowFast-like models:

| Dimension | Candidate values |
|---|---|
| Width multiplier | small, medium, large |
| Depth multiplier | shallow, default, deep |
| Temporal kernel | short, medium, long |
| Frame stride | 1, 2, 4 |
| Input resolution | 160, 224, 256 |
| Dropout | 0.0, 0.2, 0.5 |

Possible use:

- A side experiment or future-work paragraph.
- A stronger video baseline if reviewers object that X3D/I3D were not tuned.

Risk:

- More expensive.
- Pulls the paper away from its strongest VLM story.

## Recommended direction

Add a compact **Contribution C: Resource-aware adapter architecture search**.

Recommended scope:

1. Run the search on Baboonland first because severe imbalance makes macro-F1
   and class collapse meaningful.
2. Use Qwen2.5-VL-3B as the main searched backbone because the project already
   has B1, B2, A1, and A3 infrastructure around it.
3. Include LLaVA-OV-0.5B as a small-model Pareto point if time allows.
4. Keep the search space modest: 12-24 configurations.
5. Optimize for accuracy, macro-F1, trainable parameters, and latency.

Minimal search grid:

| Dimension | Values |
|---|---|
| Backbone | Qwen2.5-VL-3B, LLaVA-OV-0.5B |
| Frames | 4, 8 |
| LoRA rank | 8, 16 |
| Loss | CE, inv-sqrt CE |
| Target modules | attention only, attention + MLP |

This is 2 x 2 x 2 x 2 x 2 = 32 configurations if fully crossed. To reduce
cost, start with Qwen2.5-VL-3B only, giving 16 configurations, then add
selected LLaVA-OV-0.5B candidates for the Pareto frontier.

## How to write it in the paper

Possible section title:

> Resource-Aware Adapter Architecture Search

Possible method paragraph:

> Rather than hand-selecting a single LoRA recipe, we define a compact
> NAS-style search space over the components that determine VLM behavior
> adaptation: temporal input length, adapter rank, adapter placement, and
> class-imbalance correction. We evaluate candidates under a multi-objective
> criterion that prioritizes validation accuracy and macro-F1 while tracking
> trainable parameter count and inference latency.

Possible result paragraph:

> The search shows that class-aware adapter design can matter as much as
> backbone scale under severe class imbalance. In particular, inv-sqrt
> reweighting consistently improves macro-F1 without adding inference cost,
> while larger frame counts improve transfer more reliably than structured
> output prompts. The resulting Pareto frontier identifies compact recipes that
> approach or exceed larger hand-picked models.

Possible contribution statement:

> We introduce a resource-aware adapter architecture search formulation for
> wildlife VLM adaptation, showing that compact searched recipes can improve
> minority-class recognition and expose accuracy-cost Pareto frontiers.

## Suggested experiments

### Minimum viable NAS experiment

Dataset:

- Baboonland train/val.

Backbone:

- Qwen2.5-VL-3B.

Grid:

| Frames | Rank | Loss | Target modules |
|---:|---:|---|---|
| 4, 8 | 8, 16 | CE, inv-sqrt CE | attention only, attention + MLP |

Metrics:

- Accuracy.
- Macro-F1.
- Per-class F1.
- Prediction distribution entropy.
- Trainable parameters.
- Inference latency.

Expected value:

- Makes B2 look systematic rather than hand-picked.
- Tests whether rank/frame/module placement tradeoffs matter.
- Creates a real Pareto frontier figure.

### Stronger experiment

Add:

- KABR validation.
- Joint KABR + Baboonland training.
- A small backbone such as LLaVA-OV-0.5B.

This would connect NAS-style search to Contribution A:

> Does the searched recipe transfer from in-domain Baboonland to joint
> cross-dataset training?

## Cautions

- Do not claim to search the VLM backbone architecture unless the experiment
  actually modifies backbone internals.
- If the search is a grid or random search, say that directly.
- If no controller, predictor, or evolutionary loop is used, avoid presenting it
  as full NAS in the classic sense.
- The strongest honest terms are:
  - NAS-style adapter search
  - resource-aware adapter architecture search
  - neural pipeline search
  - parameter-efficient adaptation search

## Best one-sentence version

> We add a compact NAS-style search over temporal input design, LoRA adapter
> topology, and class-aware losses, showing that resource-aware adapter
> architecture search improves minority-class behavior recognition and reveals
> accuracy-cost Pareto frontiers for wildlife VLMs.

---

# Elaborated NAS Ideas

This section expands each brainstormed idea into a fuller research path. The
goal is to make clear what each idea would add scientifically, what experiments
would be needed, how hard it would be, and how it would fit into the current
paper.

## 1. Adapter architecture search

### Basic idea

The base VLM is fixed, but the LoRA adapter design is treated as the searchable
architecture. Instead of hand-picking one adapter recipe, search over where the
adapter is inserted, how large it is, and how it is trained.

This is the cleanest NAS-style idea for the current project because the model
being searched is small enough to train repeatedly. The search is not over the
entire neural network. It is over the parameter-efficient architecture attached
to the VLM.

### Why it is scientifically meaningful

LoRA is often treated as a default setting: choose a rank, choose target
modules, train. But those choices are architectural. A rank-8 attention-only
adapter and a rank-16 attention+MLP adapter are different trainable networks
with different parameter counts, memory use, and adaptation capacity.

For wildlife behavior recognition, this matters because the task has two
competing requirements:

- The model must adapt enough to learn domain-specific behavior labels.
- The adapter should stay small enough to remain practical and avoid overfitting
  small or imbalanced datasets.

This idea asks:

> What is the smallest or cheapest adapter architecture that still gives strong
> behavior recognition performance?

### Concrete search space

| Search variable | Why it matters |
|---|---|
| LoRA rank | Controls adapter capacity and trainable parameter count |
| Target modules | Determines whether adaptation happens in attention, MLPs, or both |
| Layer coverage | Tests whether all layers are needed or only higher semantic layers |
| LoRA dropout | Regularizes adapter training |
| Loss type | Connects adapter design to class-collapse mitigation |
| Frames per clip | Changes the visual evidence available to the adapter |

Minimal version:

| Variable | Values |
|---|---|
| Rank | 8, 16 |
| Target modules | attention only, attention + MLP |
| Loss | CE, inv-sqrt CE |
| Frames | 4, 8 |

That gives 16 Qwen2.5-VL-3B runs if fully crossed.

### What experiments to run

Start with Baboonland because it is the harder and more imbalanced dataset.

Primary experiment:

- Backbone: Qwen2.5-VL-3B.
- Dataset: Baboonland.
- Training: 1 epoch LoRA, same split and evaluation as existing B1/B2.
- Grid: frames x rank x target modules x loss.

Secondary experiment if time allows:

- Take the top 2-3 adapter recipes from Baboonland.
- Run them on KABR or joint KABR+Baboonland.
- Test whether the searched recipe transfers across datasets.

### What results would be interesting

Possible outcomes:

- Rank 8 performs almost as well as rank 16, giving a cheaper adapter.
- Attention-only LoRA is enough, reducing trainable parameters.
- Attention+MLP improves macro-F1, suggesting behavior recognition needs more
  than attention retuning.
- Inv-sqrt loss helps regardless of adapter topology.
- The best macro-F1 recipe is not the best accuracy recipe, producing a real
  accuracy/F1 tradeoff.

### Paper framing

This could become:

> Contribution C: Resource-aware adapter architecture search.

Claim:

> Searching over LoRA adapter topology identifies compact adapters that preserve
> accuracy while improving macro-F1 and reducing trainable parameter count.

### Risks

- If all adapter choices perform similarly, the result may look weak.
- If the best recipe is simply the existing rank-16 setup, the search mostly
  just validates the current design.
- Reviewers may call it hyperparameter search unless the paper clearly argues
  that LoRA rank, placement, and layer coverage define an adapter architecture.

### Difficulty

Medium. This requires new training runs but minimal code changes if the existing
config system already exposes LoRA rank and target modules. It may require small
config additions for attention-only versus attention+MLP targeting.

## 2. Temporal input architecture search

### Basic idea

A VLM does not ingest video in the same way a video CNN does. In this project,
the clip is represented as a set of sampled frames. That frame representation is
effectively the VLM's temporal input architecture.

This idea searches how the video clip should be converted into VLM input.

### Why it is scientifically meaningful

Animal behavior is temporal. A single frame may not distinguish standing,
starting to walk, vigilance, grooming, or foraging. The number and placement of
frames can determine whether the model sees the behavior or misses it.

The current project already has evidence that temporal input matters:

- 8-frame Qwen transfer performs better than 4-frame transfer.
- Structured-output prompts performed worse in transfer than plain 8-frame
  prompting.

This suggests that the clip representation itself may matter more than prompt
format.

### Concrete search space

| Search variable | Why it matters |
|---|---|
| Number of frames | More frames provide more motion evidence but cost more compute |
| Sampling strategy | Uniform sampling may miss brief behavior; motion-aware sampling may help |
| Clip span | Full-clip sampling vs centered sampling changes temporal context |
| Frame ordering | Chronological order tests whether the VLM uses temporal sequence cues |
| Prompt layout | Direct multi-image prompts vs grid images vs per-frame summaries |

Minimal version:

| Variable | Values |
|---|---|
| Frames | 4, 8, 12 |
| Sampling | uniform, center-biased |
| Frame order | chronological, shuffled control |

### What experiments to run

Low-cost evaluation-first version:

- Use existing trained adapters where possible.
- Evaluate different frame counts or sampling strategies without retraining, if
  the eval code supports changing frame sampling.

Training version:

- Train separate LoRA adapters with 4, 8, and 12 frames.
- Evaluate each in-domain and cross-dataset.
- Compare whether the best frame count differs for in-domain performance and
  transfer performance.

Control experiment:

- Evaluate chronological frames versus shuffled frames.
- If shuffled frames perform similarly, the VLM may be relying more on static
  appearance than temporal sequence.
- If chronological frames perform better, the VLM is using some temporal
  information from the ordered image sequence.

### What results would be interesting

Possible outcomes:

- 8 frames is the best tradeoff between performance and latency.
- 12 or 16 frames improves macro-F1 for rare behavior classes but costs more.
- Motion-aware sampling improves rare or brief behaviors.
- Shuffling frames hurts, suggesting temporal order matters.
- Shuffling frames does not hurt, suggesting the model is mostly doing
  appearance-based classification.

### Paper framing

Claim:

> Temporal input architecture is a first-order design variable for VLM wildlife
> behavior recognition; searched frame representations improve transfer more
> reliably than prompt formatting.

This would strengthen the paper because it explains why VLMs can or cannot act
as video behavior models despite being image-language models.

### Risks

- Changing frame count may require retraining because the training prompt and
  input shape are part of the learned behavior.
- More frames can hit memory limits or slow evaluation.
- If performance saturates at 8 frames, the result is useful but not dramatic.

### Difficulty

Low to medium. Frame count and sampling are likely easier to vary than model
internals. Motion-aware sampling would require more implementation work unless
simple frame-difference heuristics are enough.

## 3. Search class-collapse mitigation

### Basic idea

Class collapse is the failure mode where the model overpredicts common
behaviors and ignores rare ones. This idea searches the training recipe that
best prevents collapse.

This is the most direct extension of Contribution B.

### Why it is scientifically meaningful

The project already shows that zero-shot VLMs can have deceptively high
accuracy but poor macro-F1 because they overpredict majority classes. Baboonland
is especially useful here because it has severe class imbalance.

The current B2 result shows that inv-sqrt class weighting helps. A search would
make this more systematic:

> Instead of trying one hand-picked reweighting rule, search over the strength
> and form of imbalance correction.

### Concrete search space

| Search variable | Values | Meaning |
|---|---|---|
| Weight exponent alpha | 0.0, 0.25, 0.5, 0.75, 1.0 | How aggressively rare classes are upweighted |
| Focal gamma | 1.0, 2.0, 3.0 | How strongly easy examples are downweighted |
| Label smoothing | 0.0, 0.05, 0.1 | How much to soften target labels |
| Sampler strength | none, mild, full | How aggressively to rebalance batches |
| LoRA rank | 8, 16 | Whether imbalance correction interacts with adapter capacity |

The class-weight rule could be generalized as:

```text
weight(class) = count(class)^(-alpha)
```

where:

- `alpha = 0.0` is no class weighting.
- `alpha = 0.5` is inverse-square-root weighting.
- `alpha = 1.0` is inverse-frequency weighting.

### What experiments to run

Minimum experiment:

- Backbone: Qwen2.5-VL-3B.
- Dataset: Baboonland.
- Fixed LoRA setup: current rank and target modules.
- Search only alpha: 0.0, 0.25, 0.5, 0.75, 1.0.

Better experiment:

- Cross alpha with rank: 8 and 16.
- Optionally include label smoothing.

Metrics:

- Accuracy.
- Macro-F1.
- Per-class F1.
- Prediction distribution entropy.
- Number of predicted classes used.
- Majority-class prediction rate.

### What results would be interesting

Possible outcomes:

- `alpha = 0.5` is best, validating inv-sqrt weighting.
- `alpha = 0.75` improves macro-F1 but hurts accuracy.
- `alpha = 1.0` overcorrects and destabilizes training.
- Class weighting increases the number of predicted classes.
- The best alpha differs between KABR and Baboonland, showing imbalance severity
  should control search strength.

### Paper framing

Claim:

> Class-collapse mitigation can be formulated as a small architecture/training
> search over imbalance correction strength. The optimal recipe balances
> minority-class recall against majority-class accuracy.

This would turn the B2 result from "we tried a good loss" into "we searched a
family of anti-collapse methods and found a robust tradeoff."

### Risks

- This is closer to hyperparameter search than NAS.
- Too many variants could clutter the paper.
- If the best result is only marginally better than B2, the search may belong in
  an appendix.

### Difficulty

Low. The project already has class-aware training infrastructure. The main work
would be adding a continuous or discrete alpha option if it does not already
exist.

## 4. Multi-objective NAS / Pareto frontier

### Basic idea

Instead of optimizing only accuracy, search for configurations that balance
accuracy, macro-F1, model size, trainable parameters, inference latency, and
memory use.

This is the most deployment-oriented NAS idea.

### Why it is scientifically meaningful

Wildlife monitoring is often resource-constrained. A model that is 1% less
accurate but much faster or smaller may be preferable for field deployment or
large-scale video processing.

The current project already has ingredients for a Pareto story:

- Multiple VLM sizes: 0.5B, 1.7B, 3B, 4B, 7B.
- Specialist video baselines.
- LoRA versus zero-shot versus joint LoRA.
- Latency benchmark artifacts.
- Different frame counts.

This idea asks:

> Which configurations are non-dominated when considering both performance and
> cost?

A configuration is Pareto-efficient if no other configuration is better on all
important metrics.

### Concrete objective

Track:

- Accuracy.
- Macro-F1.
- Trainable parameter count.
- Total parameter count.
- Inference latency per clip.
- GPU memory.
- Frames per clip.
- Training time.

Possible Pareto comparisons:

- Macro-F1 versus latency.
- Accuracy versus trainable parameters.
- Macro-F1 versus GPU memory.
- Accuracy versus frames per clip.

### What experiments to run

This can reuse many existing results.

Step 1:

- Collect existing metrics into a single table.
- Include model, adaptation method, frame count, accuracy, macro-F1, and
  latency if available.

Step 2:

- Add missing latency measurements for the strongest configurations.
- Add trainable parameter counts for LoRA adapters.

Step 3:

- Plot Pareto frontier:
  - x-axis: latency or trainable parameters.
  - y-axis: macro-F1 or accuracy.

Step 4:

- Optionally run a few extra targeted configurations to fill gaps:
  - Qwen3B rank 8.
  - Qwen3B 4-frame inv-sqrt.
  - LLaVA-0.5B inv-sqrt if supported.

### What results would be interesting

Possible outcomes:

- LLaVA-0.5B is a strong low-cost Pareto point.
- Qwen3B joint LoRA is the best balanced point.
- 7B models improve accuracy but are dominated on cost.
- X3D-L remains latency-efficient but loses the accuracy frontier.
- Inv-sqrt loss moves a point upward in macro-F1 without moving right in
  latency, making it a pure Pareto improvement.

### Paper framing

Claim:

> Resource-aware search reveals that the best wildlife behavior model is not
> simply the largest VLM; smaller adapted VLMs and class-aware adapters occupy
> strong accuracy-cost Pareto points.

This would be one of the most practical contributions because it helps users
choose a model based on their compute budget.

### Risks

- Latency measurements must be standardized carefully.
- Different hardware or batch sizes can change latency conclusions.
- Reviewers may ask whether the search was truly automatic or simply
  retrospective benchmarking.

### Difficulty

Medium-low if latency artifacts are already reliable. Medium if latency needs
to be rerun under standardized conditions.

## 5. Neural pipeline search over model families

### Basic idea

The search object is the entire behavior-recognition pipeline rather than just
the adapter. The pipeline includes the backbone, prompting style, adaptation
method, frame count, and dataset recipe.

This is broader than adapter search. It is closer to AutoML or neural pipeline
search.

### Why it is scientifically meaningful

In practice, a researcher or conservation team does not only ask:

> What LoRA rank should I use?

They ask:

> Which model family, adaptation method, prompt, and frame count should I use
> for this dataset and compute budget?

The current project already evaluates many pieces of this pipeline. Reframing
them as a search space could make the paper feel more systematic.

### Concrete search space

| Search variable | Values |
|---|---|
| Backbone | Qwen3B, Qwen7B, LLaVA0.5B, LLaVA7B, Gemma4B, MobileVLM |
| Adaptation | zero-shot, in-domain LoRA, class-aware LoRA, joint LoRA |
| Frames | 4, 8 |
| Prompt style | plain label, structured output |
| Training data | KABR only, Baboonland only, joint |
| Objective | accuracy, macro-F1, latency, transfer |

### What experiments to run

This could start as a meta-analysis of existing experiments:

- Build a unified results table.
- Mark each row as a pipeline configuration.
- Identify the best pipeline for each objective:
  - Best KABR accuracy.
  - Best Baboonland macro-F1.
  - Best transfer.
  - Best small-model result.
  - Best accuracy-cost tradeoff.

Additional experiments could be minimal:

- Fill only missing cells needed to make comparisons fair.
- Avoid creating a massive grid across all backbones.

### What results would be interesting

Possible outcomes:

- The best zero-shot model is not the best LoRA model.
- Larger models help transfer, but small models can be surprisingly strong
  in-domain.
- Joint training beats transfer-only pipelines.
- Prompt engineering is less important than adaptation and frame count.
- Class-aware loss helps most on the imbalanced Baboonland dataset.

### Paper framing

Claim:

> A pipeline-level search shows that wildlife behavior recognition depends on
> interactions among backbone, adaptation, temporal input, and imbalance
> correction; no single axis such as model scale explains performance.

This could be valuable as an organizing framework for the whole paper.

### Risks

- This is the easiest idea for reviewers to dismiss as "just benchmarking."
- It may weaken the NAS language if not paired with a real search procedure.
- The search space may become too broad to cover cleanly.

### Difficulty

Low if presented as an analysis of existing results. Medium if turned into a
formal experiment with missing configurations filled in.

## 6. NAS for specialist video baselines

### Basic idea

Apply a more traditional NAS or architecture search to the video baselines,
such as X3D or SlowFast, rather than to the VLM adapters.

This would search the architecture of a video classifier: width, depth, temporal
kernel, frame stride, resolution, and regularization.

### Why it is scientifically meaningful

One possible reviewer objection is:

> The VLMs beat X3D/I3D, but were the video baselines tuned well enough?

A small search over video model architectures could strengthen the comparison
by showing that adapted VLMs still beat a more carefully tuned specialist video
baseline.

This is also the most traditional interpretation of NAS, because it searches
actual neural network architecture parameters.

### Concrete search space

| Search variable | Why it matters |
|---|---|
| Width multiplier | Controls model capacity |
| Depth multiplier | Controls network depth |
| Temporal kernel | Controls temporal modeling range |
| Frame stride | Controls temporal resolution |
| Input resolution | Controls spatial detail and compute |
| Dropout | Controls regularization |
| Learning rate | Training stability and final accuracy |

### What experiments to run

Minimum defensive experiment:

- Pick one video family, probably X3D.
- Search 6-12 configurations on Baboonland.
- Keep training budget fixed.
- Compare best searched X3D to adapted VLMs.

Possible configurations:

| Candidate | Width | Depth | Frames | Resolution | Dropout |
|---|---|---|---:|---:|---:|
| X3D-small-fast | small | shallow | 8 | 160 | 0.2 |
| X3D-default | default | default | 16 | 224 | 0.5 |
| X3D-wide | wide | default | 16 | 224 | 0.5 |
| X3D-long | default | default | 32 | 224 | 0.5 |
| X3D-highres | default | default | 16 | 256 | 0.5 |

### What results would be interesting

Possible outcomes:

- Tuned X3D improves over current X3D-L but still trails adapted VLMs.
- Tuned X3D becomes more competitive on KABR but not Baboonland.
- Video models remain faster but less accurate.
- Video models have better latency but worse transfer/generalization.

### Paper framing

Claim:

> Even after a compact architecture search over specialist video baselines,
> adapted VLMs retain the highest accuracy, while video models remain attractive
> on the latency frontier.

This would make the VLM comparison harder to dismiss.

### Risks

- This is expensive compared with adapter search.
- It may pull the paper away from the VLM contribution.
- It requires careful training fairness.
- If searched X3D gets much better, the headline "VLMs beat video specialists"
  may weaken, though the paper could still be stronger and more honest.

### Difficulty

High. This is the most compute-heavy option and likely the least necessary
unless reviewers specifically challenge the specialist video baselines.

## Overall ranking

| Idea | Scientific fit | Compute cost | Paper value | Risk | Recommendation |
|---|---|---:|---:|---:|---|
| Adapter architecture search | High | Medium | High | Medium | Best main NAS direction |
| Temporal input architecture search | High | Low-medium | High | Low-medium | Strong and domain-relevant |
| Class-collapse mitigation search | Very high | Low | Medium-high | Low | Best extension of Contribution B |
| Multi-objective Pareto search | High | Low-medium | High | Medium | Best deployment framing |
| Pipeline search over model families | Medium | Low | Medium | Medium-high | Useful framing, weaker NAS claim |
| Specialist video NAS | Medium | High | Medium | High | Best as optional reviewer defense |

## Recommended combined version

The strongest version is not to choose only one idea. It is to combine the
lowest-risk pieces into one compact contribution:

> **Resource-aware adapter and temporal architecture search.**

Search over:

- Frames: 4 vs 8.
- LoRA rank: 8 vs 16.
- Target modules: attention only vs attention+MLP.
- Loss: CE vs inv-sqrt CE.

Evaluate:

- Accuracy.
- Macro-F1.
- Per-class F1.
- Prediction distribution entropy.
- Trainable parameters.
- Latency.

Primary dataset:

- Baboonland, because it stresses class imbalance.

Optional transfer check:

- Apply the best recipe to joint KABR+Baboonland training.

Best paper claim:

> A compact resource-aware search over adapter topology, temporal input, and
> class-aware loss identifies Pareto-efficient VLM adaptation recipes, showing
> that minority-class behavior recognition can be improved without increasing
> inference cost.

This combines the strengths of ideas 1, 2, 3, and 4 while avoiding the cost and
messiness of a full model-family or video-model NAS study.
