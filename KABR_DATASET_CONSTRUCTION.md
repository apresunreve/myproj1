# KABR Dataset Construction For The Current Experiments

This note explains how the current KABR dataset used in this repo is built for
the **clip-level behavior recognition** experiments.

It covers:

- the original KABR inputs we start from
- what is preserved from the release
- how the current derived clips are constructed
- how species and behavior labels are assigned
- how those clips are turned into model inputs for VLMs and X3D

## 1. Original KABR Inputs

The current workflow starts from the released KABR frame annotations and image
files:

- annotations:
  - `/home/jhyan/KABR_files/annotation/train.csv`
  - `/home/jhyan/KABR_files/annotation/val.csv`
- class mapping:
  - `/home/jhyan/KABR_files/annotation/classes.json`
- images:
  - `/home/jhyan/KABR_files/dataset/image`

The per-frame annotation reader is implemented in:

- [kabr_frames.py](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)

Each input frame record contains:

- `original_video_id`
- `video_id`
- `frame_id`
- `rel_path`
- `image_path`
- `label_id`
- `label_name`
- inferred `species`
- `split`

## 2. What Is Preserved

The current experiments do **not** change the underlying KABR supervision.

What stays unchanged:

- no relabeling
- no train/val split change
- no new human annotation
- no frame removal from the released split files

So the current setup is a **derived representation** of KABR, not a relabeled
or resplit dataset.

## 3. Species Inference

Species are inferred from the `original_video_id` prefix:

- `G...` -> `giraffe`
- `ZP...` -> `plains zebra`
- `ZG...` -> `grevys zebra`

This logic is in:

- [infer_species](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)

## 4. Derived Clip Construction

The original KABR release is frame-labeled. The current clip-level setup turns
those frame labels into single-label clips.

One derived clip is defined as a **contiguous same-label run** satisfying all of
the following:

- same `original_video_id`
- same `label_id`
- consecutive `frame_id`

A new derived clip starts when any of these changes:

- `original_video_id`
- `label_id`
- frame continuity

The exact grouping code is:

- [group_frame_records_into_clips](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)

So if a sequence looks like this:

- frames `1..90` = `Head Up`
- frames `91..140` = `Walk`

then it becomes:

- one `Head Up` clip
- one `Walk` clip

Each derived clip therefore has exactly **one behavior label**.

## 5. Derived Clip Fields

Each derived clip stores:

- `clip_id`
- `original_video_id`
- `video_id`
- `species`
- `label_id`
- `label_name`
- `frame_ids`
- `rel_paths`
- `image_paths`
- `split`

The clip id format is:

- `original_video_id:start_frame-end_frame:label_id`

Example shape:

- `G0068.1:1-90:3`

## 6. Current Derived Clip Counts

On the current local KABR copy, the derived clip counts are:

- train: `11003`
- val: `3761`

These are the clip units used by the current behavior experiments.

The clip manifests used by the clip-level X3D path are in:

- `/home/jhyan/KABR_files/annotation/clip_manifests`

There is also a tiny smoke version used for debugging:

- `/home/jhyan/KABR_files/annotation/clip_manifests_smoke`

## 7. Behavior Labels

The current behavior experiments use the 8 KABR behavior classes:

- `Walk`
- `Graze`
- `Browse`
- `Head Up`
- `Auto-Groom`
- `Trot`
- `Run`
- `Occluded`

Each derived clip has exactly one of these labels.

## 8. How VLM Inputs Are Built

The VLMs do **not** consume raw video files.

Instead, each derived clip is converted into:

- a small ordered set of RGB frames
- plus a text prompt

Default VLM clip representation:

- sample up to `8` frames from each derived clip
- keep the sampled frames in temporal order
- use a prompt asking the model to output exactly one behavior label

Frame sampling is uniform over the clip:

- if the clip already has `<= max_frames`, keep all frames
- otherwise choose evenly spaced positions from the first to the last frame

This is implemented in:

- [sample_clip_frames](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)

That means the VLM input is:

- `8` ordered clip frames by default
- not the full raw clip frame-by-frame
- not a video tensor

There is also a `4`-frame LoRA ablation using the same derived clips.

## 9. How X3D Inputs Are Built

X3D uses the **same derived clip boundaries**, but it does **not** use the VLM’s
exact sampled frame list.

Instead:

- one derived clip is the source unit
- X3D samples a fixed temporal tensor from inside that clip
- short clips are handled by the SlowFast clip loader logic

So X3D and the VLM share:

- the same clip ids
- the same train/val clip split
- the same behavior labels

But they differ in the final input representation:

- VLM: ordered sampled still images + prompt
- X3D: fixed video tensor sampled from inside the clip

## 10. Why This Representation Exists

This derived clip-level representation was introduced to make comparisons
fairer across model families.

It gives us:

- the same clip-level evaluation unit for VLMs and X3D
- a direct way to compare:
  - zero-shot VLMs
  - post-trained VLMs
  - supervised X3D baselines

Without this step, the released KABR frame annotations and the original
Charades-style/SlowFast training path would not line up cleanly with the
clip-based VLM evaluation workflow.

## 11. Important Caveat

This current setup is a **derived clip-level benchmark built from KABR frame
annotations**.

It is close to the original KABR task, but it is not identical to every
published KABR protocol. In particular:

- the current VLM experiments use sampled still frames from derived clips
- the current clip-level X3D experiments use the custom `kabr_clips` path
- this is different from the original Charades-style KABR training path used in
  the upstream SlowFast repo

So when comparing against published KABR numbers, the fairest claim is:

- these are results on the **matched derived clip-level KABR setup used in this
  repo**

## 12. Key Files

- frame/clip data logic:
  - [kabr_frames.py](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)
- current behavior summary:
  - [CURRENT_BEHAVIOR_SUMMARY.md](/home/jhyan/myproj/CURRENT_BEHAVIOR_SUMMARY.md)
- current project status:
  - [STATUS_AND_PLAN.md](/home/jhyan/myproj/STATUS_AND_PLAN.md)
