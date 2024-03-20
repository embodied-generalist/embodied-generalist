<h2 align="center">
  <span><img src="assets/leo.svg" width="4%" style="transform: translate(0,9px)"></span><b>An Embodied Generalist Agent in 3D World</b>
</h2>

<div align="center" margin-bottom="6em">
<a target="_blank" href="https://huangjy-pku.github.io/">Jiangyong Huang<sup>✶</sup></a>,
<a target="_blank" href="https://silongyong.github.io/">Silong Yong<sup>✶</sup></a>,
<a target="_blank" href="https://jeasinema.github.io/">Xiaojian Ma<sup>✶</sup></a>,
<a target="_blank" href="https://github.com/Germany321">Xiongkun Linghu<sup>✶</sup></a>,
<a target="_blank" href="https://xiaoyao-li.github.io/">Puhao Li</a>,
<br/>
<a target="_blank" href="https://github.com/jetpackfirstme">Yan Wang</a>,
<a target="_blank" href="https://liqing-ustc.github.io/">Qing Li</a>,
<a target="_blank" href="http://www.stat.ucla.edu/~sczhu/">Song-Chun Zhu</a>,
<a target="_blank" href="https://buzz-beater.github.io/">Baoxiong Jia</a>,
<a target="_blank" href="https://siyuanhuang.com/">Siyuan Huang</a>

</div>
&nbsp;

<div align="center">
    <a href="https://arxiv.org/abs/2311.12871" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://embodied-generalist.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Page-LEO-9cf" alt="Project Page"></a>
    <a href="https://youtu.be/mlnjz4eSjB4?si=NN9z7TpkTPgBAzBw" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-9966ff" alt="Video"></a>
    <a href="https://drive.google.com/drive/folders/1dko2dzdwRWSK3hi1liBpGHZ8Dz97jXdP?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/Data-LEO-blue" alt="Data"></a>
    <a href="placeholder" target="_blank">
    <img src="https://img.shields.io/badge/Model-LEO-darkorange" alt="Model"></a>
</div>
&nbsp;

<div align="left">
<img src="assets/teaser.png" width="99%" alt="LEO Teaser">
</div>

We introduce **LEO**, an **embodied multi-modal generalist agent** capable of **grounding**, **reasoning**, **chatting**, **planning**, and **acting** in the **3D world**. **LEO** is trained in a two-stage scheme: *(i) 3D vision-language (VL) alignment* and *(ii) 3D vision-language-action (VLA) instruction tuning*.

We meticulously collect extensive diverse data for training **LEO**. <sup>&dagger;</sup> indicates the task contains our generated data. See [Task and Data](#task-and-data) for details. We show the data statistics as below:

| Dataset | Task | 2D required? | 3D assets | #data |
| :---: | :---: | :---: | :---: | :---: |
| *LEO-align* | object captioning | ✗ | Objaverse | 660k |
|  | object referring<sup>&dagger;</sup> | ✗ | ScanNet + 3RScan | 354k |
|  | scene captioning<sup>&dagger;</sup> | ✗ | 3RScan | 20k |
| *LEO-instruct* | 3D captioning | ✗ | ScanNet | 37k |
|  | 3D QA<sup>&dagger;</sup> | ✗ | ScanNet + 3RScan | 83k |
|  | 3D dialogue<sup>&dagger;</sup> | ✗ | 3RScan | 11k |
|  | task planning<sup>&dagger;</sup> | ✗ | 3RScan | 14k |
|  | navigation | ✓ | MP3D | 60k |
|  | manipulation | ✓ | CLIPort | 300k |


## News
**[2024.03]** We release the code, data and model weights. The embodied AI (EAI) tasks (navigation and manipulation) need further organization and will be released soon.

**[2024.01]** We release a [Huggingface interactive demo](https://huggingface.co/spaces/embodied-generalist/LEO-Demo). Chat with **LEO** and enjoy yourself.

## Get Started

1. Clone Github repo.
```shell
git clone git@github.com:embodied-generalist/embodied-generalist.git
cd embodied-generalist
```

2. Create `conda` environment and install dependencies.
```shell
conda create -n leo python=3.9
conda activate leo

# install PyTorch, take our version for example
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install other dependencies with pip
pip install -r requirements.txt

# install peft separately to escape its install_requires
pip install peft==0.5.0 --no-deps
```

3. Install third party libraries (for point cloud backbones).
```shell
cd model

# default PointNet++
cd pointnetpp
python setup.py install
cd ..

# optional: PointNext (if you want to substitute the default PointNet++)
cd pointnext/cpp/pointnet2_batch
python setup.py install
cd ../../../

cd ..
# sanity check
python -c 'from model.pointnetpp.pointnetpp import PointNetPP'
# for PointNext, run 'from model.pointnext.pointnext import PointNext'
```

4. Go through [task and data](#task-and-data), [model weights](#model-weights), and you are ready to [run](#running).

## Task and Data
**Data preparation.** The [data](https://drive.google.com/drive/folders/1dko2dzdwRWSK3hi1liBpGHZ8Dz97jXdP?usp=sharing) includes two components: scan data and language annotations.
- **Scan data.** To simplify the preparation and save storage, we streamline the scan data (point clouds and instance segments), which is less than 10G yet already sufficient for experiments on **LEO**. You can download the compressed files from the links below and arrange the data according to the illustration of scan data structure.
  - **ScanNet**: [pcd_with_global_alignment](https://drive.google.com/file/d/1N9LQfDlC0UxJCPefdYeP54TT-TfwOrLh/view?usp=sharing), [mask (Mask3D proposals)](https://drive.google.com/file/d/1AACMAjlFFGZsKJYAh8hFikFGKtYC6epk/view?usp=sharing).
  - **3RScan**: [3RScan-ours-align](https://drive.google.com/file/d/11xltRZ2BVzJWN4uqOAQgbLspWjacQyAJ/view?usp=sharing).
  - **Cap3D**. Please refer to [Cap3D data](https://huggingface.co/datasets/tiange/Cap3D) for preparing the point clouds, where we use [pcs_pt](https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips). The corresponding annotation file (`Cap3D_automated_Objaverse_no3Dword.csv`) is included in our released annotations.
```
# scan data structure

├── ${scannet_base}
    ├── scan_data
    │   └── pcd_with_global_alignment
    │       ├── ${scene_id}.pth
    └── mask
        ├── ${scene_id}.mask.npz

├── ${rscan_base}
    └── 3RScan-ours-align
        ├── ${scene_id}
            ├── pcd-align.pth
            └── inst_to_label.pth

├── ${cap3d_root}
    ├── Cap3D_pcs_pt
    │   ├── ${obj_id}.pt
    └── Cap3D_automated_Objaverse_no3Dword.csv   # included in annotations
```

- **Language annotations.** The annotations are categorized into two parts according to the training stage. We provide a [compressed file](https://drive.google.com/file/d/1ggFHOvxmGrBARqcMPcRSVmCXbsSZle2F/view?usp=sharing) that wraps up all the annotations, which should be organized in the following structure:

```
# annotations structure

├── ${alignment_base}
    ├── obj_caption -> ${cap3d_root}
    │   ├── Cap3D_pcs_pt
    │   │   ├── ${obj_id}.pt
    │   └── Cap3D_automated_Objaverse_no3Dword.csv
    ├── obj_scene_caption
    │   ├── 3rscan_prompted.json
    │   ├── 3rscan_scanscribe.json
    │   ├── scannet_referit3d_nr3d.jsonl
    │   └── scannet_referit3d_sr3d+.jsonl
    └── scene_caption
        └── 3rscan_scene_cap_prompted.json

├── ${instruction_base}
    ├── scan2cap
    │   ├── scannet_scanrefer.jsonl
    │   ├── scannetv2_train.txt
    │   └── scannetv2_val.txt
    ├── scanqa
    │   ├── ScanQA_v1.0_train.json
    │   └── ScanQA_v1.0_val.json
    ├── sqa3d
    │   ├── v1_balanced_questions_train_scannetv2.json
    │   ├── v1_balanced_questions_val_scannetv2.json
    │   ├── v1_balanced_questions_test_scannetv2.json
    │   ├── v1_balanced_sqa_annotations_train_scannetv2.json
    │   ├── v1_balanced_sqa_annotations_val_scannetv2.json
    │   └── v1_balanced_sqa_annotations_test_scannetv2.json
    ├── 3rscanqa
    │   └── 3rscan_qa_prompted_aug.json
    ├── dialogue
    │   └── 3rscan_dialog_prompted.json
    └── planning
        └── 3rscan_plan_prompted.json
```

**Data configurations.** After data preparation, check `configs/data/default.yaml` to update the paths, including `scan_family_base`, `rscan_base`, `alignment_base`
and `instruction_base`.

**Dataloaders.** The implementation of dataset per task lies in `data/datasets.py`, where `LeoMix` aggregates various datasets as the training dataset.


## Model Weights
**Pretrained weights to load.**
- **LLM**: [Vicuna-7B](https://huggingface.co/huangjy-pku/vicuna-7b/tree/main). We use Vicuna v1.1 from [FastChat](https://github.com/lm-sys/FastChat), which you can refer to for the access of Vicuna-13B or more advanced versions. Remember to update `cfg_path` in `configs/llm/*.yaml`.
- **Point cloud backbone**: [PointNet++](https://drive.google.com/file/d/1vq2ro_OwoTsa6JuIQVqfNFSbyWRuUGcG/view?usp=sharing), [PointBERT](https://drive.google.com/file/d/1RqjuUs6SI-0olklHGtCTT9h5jyV0gBxM/view?usp=sharing). We have not tried `PointNext`, but everything is ready except the pretrained weights. Remember to update `path` in `configs/vision3d/backbone/*.yaml`.

**Instruction-tuned LEO weights.** Coming soon.


## Running
The training pipeline is elaborated in `trainer/leo_trainer.py`. Make sure the config file `configs/default.yaml` is properly set up before running.
- **General setup.** We use `wandb` as the default experiment logger. Remember to modify `logger.entity` to your account and init the `wandb`. Modify `name`, `note`, and `base_dir` for proper experiment output.
- **Model.** The components of `LeoAgent` can be configured in `configs/llm`, `configs/vision2d` and `configs/vision3d`.
- **Task.** You can configure the tasks by specifying a `yaml` in `configs/task`. You can also run new tasks by creating similar configs.

We prepare some running scripts in `scripts/`, covering `train_align`, `train_tuning`, and `test`. The core is to run `launch.py` with proper arguments. There are three launch modes:
```shell
# python launch
python launch.py --mode python --config configs/default.yaml <HYDRA_CONFIG>

# accelerate launch
python launch.py --mode accelerate --config configs/default.yaml <HYDRA_CONFIG>

# SLURM submitit launch, default
python launch.py --mode submitit --config configs/default.yaml <HYDRA_CONFIG>

# for example, run alignment with submitit
python launch.py --mode submitit \
                 --config configs/default.yaml \
                 --name leo_tuning \ # job name
                 --qos lv0b \   # QoS
                 --time 48 \   # job execution duration (hour)
                 --num_nodes 1 \
                 --partition HGX \   # node type
                 --gpu_per_node 4 \
                 --mem_per_gpu 100 \   # memory per GPU
                 --port 2050 \
                 task=align \   # hydra: cfg.task, select task
                 note=align_lora \   # hydra: cfg.note, for exp_dir
```

For more arguments, use `python launch.py --help`. Refer to [SLURM submitit](https://github.com/facebookincubator/submitit) or [Accelerate](https://huggingface.co/docs/accelerate/index) for more information.


## Notes
We manually modify some methods of `accelerate.Accelerator` in `common/misc.py`, including `gather_for_metrics` (fix gathering non-tensor objects), `get_state_dict` (for saving only learnable parameters when calling `save_state`), and `prepare_scheduler` (fix behavior with gradient accumulation).


## BibTex
```bibtex
@article{huang2023embodied,
  title={An Embodied Generalist Agent in 3D World},
  author={Huang, Jiangyong and Yong, Silong and Ma, Xiaojian and Linghu, Xiongkun and Li, Puhao and Wang, Yan and Li, Qing and Zhu, Song-Chun and Jia, Baoxiong and Huang, Siyuan},
  journal={arXiv preprint arXiv:2311.12871},
  year={2023}
}
```
