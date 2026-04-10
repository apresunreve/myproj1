# Related Work: Wildlife Behavior Recognition, VLMs, and Transfer

This note is a focused literature survey for the current paper direction:

- wildlife behavior recognition from video or frame sequences
- zero-shot and post-trained vision-language models
- comparison against specialist video models such as X3D / SlowFast
- transfer across datasets rather than single-benchmark wins

The most useful framing is not just "VLMs for animals," but:

- what prior work exists on animal behavior understanding,
- what prior work exists on wildlife-domain foundation models,
- and where our current KABR + baboonland study would sit.

## 1. Closest Directly Relevant Work

### 1.1 Fine-tuning Vision-Language Models for Animal Behavior Analysis

- Citation:
  Sepideh Mamooler, Haozhe Qi, Valentin Gabeff, Syrielle Montariol, Antoine Bosselut, Alexander Mathis.
  *Fine-tuning Vision-Language Models for Animal Behavior Analysis*. COLM 2025.
- Link:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- This is the closest paper to our current direction.
- It converts multiple animal behavior datasets into a multi-task VideoQA dataset with code-based outputs.
- It fine-tunes InternVL3-8B and reports large gains over both vision-only baselines and zero-shot VLMs.

What it contributes:

- a dataset-conversion framework for animal behavior tasks
- multi-task VLM fine-tuning across datasets
- code-based structured outputs for interpretability

Why our paper is still distinct:

- we are working on a tighter, behavior-only comparison
- we compare zero-shot, frozen features, distillation, and LoRA directly
- we have a stronger specialist-video comparison through the X3D family
- we can study cross-dataset behavior transfer explicitly with KABR and baboonland

Bottom line:

- this is the main paper we need to position against
- it pushes the field toward post-trained VLMs for animal behavior
- our best contribution is to be sharper on adaptation-method comparisons and transfer

## 2. Dataset Papers We Should Anchor To

### 2.1 KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition From Drone Videos

- Citation:
  Maksim Kholiavchenko et al.
  *KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition From Drone Videos*. WACV Workshops 2024.
- Link:
  https://openaccess.thecvf.com/content/WACV2024W/CV4Smalls/papers/Kholiavchenko_KABR_In-Situ_Dataset_for_Kenyan_Animal_Behavior_Recognition_From_Drone_WACVW_2024_paper.pdf

Why it matters:

- This is the source dataset paper for our KABR experiments.
- It establishes the drone-video wildlife setting and the original behavior labels.

How we use it:

- as the citation for KABR data origin
- as the baseline dataset context for our derived clip-level setup

### 2.2 BaboonLand Dataset: Tracking Primates in the Wild and Automating Behaviour Recognition From Drone Videos

- Citation:
  Isla Duporge et al.
  *BaboonLand dataset: Tracking primates in the wild and automating behaviour recognition from drone videos*. IJCV 2025.
- Evidence:
  cited in the COLM 2025 animal-behavior VLM paper reference list
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- This appears to be the right paper family for the `baboonland` dataset we are now processing locally.
- It gives us the second dataset anchor for a cross-dataset story.

Why it is valuable for our paper:

- it keeps the viewpoint consistent with KABR: drone-based wildlife video
- but changes species, label space, and behavior semantics
- that makes it a useful transfer stress test

Note:

- I found a reliable citation trail through the COLM paper, but I have not yet pulled the full IJCV paper page directly.
- For the final manuscript, we should resolve and cite the canonical journal page directly.

## 3. Broader Animal Behavior Video Benchmarks

### 3.1 MammalNet: A Large-Scale Video Benchmark for Mammal Recognition and Behavior Understanding

- Citation:
  Jun Chen et al.
  *MammalNet: A Large-scale Video Benchmark for Mammal Recognition and Behavior Understanding*. CVPR 2023.
- Evidence:
  cited in the COLM 2025 paper with benchmark table entries
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- MammalNet is one of the main specialist animal-video benchmarks in the literature.
- It is useful for placing our paper in the larger space of animal video understanding.

What it means for us:

- it supports the claim that specialist supervised video benchmarks already exist
- our novelty is not "animal video exists" but how VLMs behave under zero-shot, post-training, and transfer

### 3.2 Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding

- Citation:
  Xun Long Ng et al.
  *Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding*. CVPR 2022.
- Evidence:
  cited in the COLM 2025 paper
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- Animal Kingdom is a large, broad animal-behavior benchmark.
- It is one of the standard references showing that animal behavior understanding is now a serious computer vision problem, not just a niche application.

### 3.3 LOTE-Animal: A Long Time-Span Dataset for Endangered Animal Behavior Understanding

- Citation:
  Dan Liu et al.
  *LOTE-Animal: A Long Time-Span Dataset for Endangered Animal Behavior Understanding*. ICCV 2023.
- Evidence:
  cited in the COLM 2025 paper
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- It emphasizes long-duration behavior understanding, which helps contrast our current clip-level setting.
- It supports discussion of temporal granularity as a real design choice in animal behavior work.

## 4. Recent Foundation-Model Work in Animal Behavior

### 4.1 MouseGPT: A Large-Scale Vision-Language Model for Mouse Behavior Analysis

- Citation:
  Teng Xu et al.
  *MouseGPT: A Large-scale Vision-Language Model for Mouse Behavior Analysis*. arXiv 2025.
- Link:
  https://arxiv.org/abs/2503.10212

Why it matters:

- It is a clear example of a behavior-focused VLM, rather than just wildlife species recognition.
- It shows that animal-behavior VLM work is becoming a real subfield.

How it differs from us:

- mouse/neuroscience setting instead of wildlife field video
- not obviously centered on comparison against specialist video baselines in the same way

### 4.2 AmadeusGPT: A Natural Language Interface for Interactive Animal Behavioral Analysis

- Citation:
  Shaokai Ye et al.
  *AmadeusGPT: a natural language interface for interactive animal behavioral analysis*. NeurIPS 2023.
- Link:
  https://arxiv.org/abs/2307.04858

Why it matters:

- It is one of the most visible language-model papers in animal behavior.
- It frames language models as tools for behavior analysis workflows, not just label prediction.

How it relates:

- it supports the broader motivation that language-grounded systems are useful in animal behavior research
- but it is not a direct baseline for clip-level classification

### 4.3 SuperAnimal Pretrained Pose Estimation Models for Behavioral Analysis

- Citation:
  Shaokai Ye et al.
  *SuperAnimal pretrained pose estimation models for behavioral analysis*. Nature Communications 2024.
- Link:
  https://arxiv.org/abs/2203.07436

Why it matters:

- This is not a VLM paper, but it is very relevant background.
- It represents the "foundation model for animal behavior pipelines" idea from the pose-estimation side.

Why it helps our framing:

- it shows the field is moving toward reusable animal foundation models
- our paper sits on the appearance/video-language side rather than the pose side

## 5. Zero-Shot and Evaluation-Centered Work

### 5.1 Zero-shot Animal Behaviour Classification with Vision-Language Foundation Models

- Citation:
  Simon Chamaille-Jammes.
  *Zero-shot animal behaviour classification with vision-language foundation models*. Methods in Ecology and Evolution, 2025.
- Evidence:
  cited in the COLM 2025 paper reference list
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- This is directly relevant to our zero-shot baseline story.
- It suggests there is already interest from ecology in whether foundation models can classify animal behavior without task-specific training.

Why it helps us:

- it supports the motivation for our zero-shot VLM benchmark section
- our experiments can complement it by showing when zero-shot breaks down and how post-training changes the picture

### 5.2 Video Foundation Models for Animal Behavior Analysis

- Citation:
  David Ross et al.
  *Video foundation models for animal behavior analysis*. bioRxiv 2024.
- Evidence:
  cited in the COLM 2025 paper reference list
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- This appears to be a relevant bridge between general video foundation models and animal behavior analysis.
- It is likely useful as prior art for transfer and few-shot/zero-shot behavior recognition.

Note:

- I have a reliable citation trail, but not yet the canonical preprint page pulled directly.
- For the final bibliography, we should resolve the exact bioRxiv entry.

## 6. Wildlife-Domain Vision-Language Adaptation Beyond Behavior

### 6.1 WildCLIP: Scene and Animal Attribute Retrieval From Camera Trap Data With Domain-Adapted Vision-Language Models

- Citation:
  Valentin Gabeff et al.
  *WildCLIP: Scene and Animal Attribute Retrieval from Camera Trap Data with Domain-Adapted Vision-Language Models*. IJCV 2024.
- Evidence:
  cited in the COLM 2025 paper reference list
- Source mention:
  https://openreview.net/pdf?id=HVQ9MA5f7j

Why it matters:

- WildCLIP is one of the clearest wildlife-domain VLM adaptation papers.
- It is not behavior recognition, but it shows that domain-adapting VLMs for wildlife is effective and active research.

### 6.2 CATALOG: A Camera Trap Language-Guided Contrastive Learning Model

- Citation:
  Marcella Santamaria et al.
  *CATALOG: A Camera Trap Language-Guided Contrastive Learning Model*. WACV 2025.
- Link:
  https://openaccess.thecvf.com/content/WACV2025/papers/Santamaria_CATALOG_A_Camera-Trap_Language-Guided_Contrastive_Learning_Model_WACV_2025_paper.pdf

Why it matters:

- This is another strong example of wildlife-domain multimodal adaptation.
- It is more about camera-trap representation learning than behavior recognition, but it helps justify the broader relevance of language-guided wildlife models.

## 7. What the Literature Seems to Say

There is a fairly clear pattern across the papers above:

1. Animal behavior understanding is now a mature computer-vision problem.
2. There is growing interest in foundation models for animal behavior, but the area is still young.
3. The most directly relevant recent work suggests that post-training VLMs can outperform zero-shot VLMs and specialist baselines on some behavior datasets.
4. Wildlife-domain adaptation papers exist, but many focus on species recognition, retrieval, or camera-trap attributes rather than clip-level behavior recognition.
5. Cross-dataset transfer remains less explored than single-dataset fine-tuning.

That last point is especially important for us.

## 8. Best Positioning for Our Paper

Given the current project, the strongest positioning is:

- We are not just adding another zero-shot wildlife benchmark.
- We are studying which adaptation methods actually make VLMs useful for wildlife behavior recognition.
- We compare them against specialist video models.
- With KABR and baboonland together, we can start to say something about transfer, not just in-domain performance.

That suggests a paper contribution framed around:

1. zero-shot wildlife behavior recognition is brittle across datasets
2. post-training method matters more than model size alone
3. LoRA-style adaptation is much stronger than weaker adaptation baselines in our setting
4. specialist video models remain strong, especially on macro-F1 and temporal behavior structure
5. cross-dataset behavior transfer is still hard and deserves explicit evaluation

## 9. Most Important Citations for the Introduction

If we need a compact first-pass bibliography, these are the highest-priority ones:

- KABR dataset paper
- BaboonLand dataset paper
- Fine-tuning Vision-Language Models for Animal Behavior Analysis
- AmadeusGPT
- MouseGPT
- MammalNet
- Animal Kingdom
- WildCLIP
- CATALOG
- SuperAnimal

## 10. Gaps We Can Still Exploit

Based on the current literature, the most promising open gaps for our paper are:

- direct comparison of zero-shot, frozen, distill, and LoRA adaptation on wildlife behavior
- explicit comparison against X3D-family specialist video models
- cross-dataset transfer between wildlife behavior datasets
- effect of temporal evidence on VLM behavior recognition
- careful negative results showing that not every post-training method helps equally

That is a stronger and more defensible story than simply claiming that one VLM does well on one dataset.
