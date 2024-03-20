import json
import os
import random

import jsonlines
import nltk
import numpy as np
import pandas as pd
import torch
from accelerate.logging import get_logger
from einops import rearrange
from scipy import sparse
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY
from .data_utils import build_rotate_mat, construct_bbox_corners, convert_pc_to_box, \
                        eval_ref_one_sample, get_sqa_question_type
from .text_pool import *

logger = get_logger(__name__)

# len(tokenized_sentence) / len(sentence)
LLAMA_TOKEN_SENT_RATIO = 0.24

LEOMIX_REQUIRED_KEYS = [
    'source',
    'prompt_before_obj',
    'prompt_middle_1',
    'prompt_middle_2',
    'prompt_after_obj',
    'obj_fts',
    # 'obj_masks',   # this is filled by dataset wrapper
    'obj_locs',
    'anchor_locs',
    'anchor_orientation',
    'img_fts',   # currently hardcode to 224x224
    'img_masks',
    'output_gt',
]


@DATASET_REGISTRY.register()
class LeoBase(Dataset):
    r""" Unified input format:
    <prompt_before_obj> + <prompt_middle_1> + <img_tokens> + <prompt_middle_2> + <obj_tokens> + <prompt_after_obj>
    <prompt_before_obj>: <role_prompt> + <situation_prompt>
    <prompt_middle_1>: <egoview_prompt> (masked if unnecessary)
    <prompt_middle_2>: <objects_prompt>
    <prompt_after_obj>: <task_prompt>
    <output_gt>: response label, will be appended to input sequence for computing loss during training
    """

    role_prompt = "You are an AI visual assistant situated in a 3D scene. "\
                  "You can perceive (1) an ego-view image (accessible when necessary) and (2) the objects (including yourself) in the scene (always accessible). "\
                  "You should properly respond to the USER's instruction according to the given visual information. "
    situation_prompt = "{situation}"
    egoview_prompt = "Ego-view image:"
    objects_prompt = "Objects (including you) in the scene:"
    task_prompt = "USER: {instruction} ASSISTANT:"

    @staticmethod
    def get_prompts(instruction, situation="", dialogue=None):
        return {
            'prompt_before_obj': LeoBase.role_prompt + LeoBase.situation_prompt.format(situation=situation),
            'prompt_middle_1': LeoBase.egoview_prompt,
            'prompt_middle_2': LeoBase.objects_prompt,
            'prompt_after_obj': LeoBase.task_prompt.format(instruction=instruction) if dialogue is None else dialogue,
        }

    @staticmethod
    def check_output_and_fill_dummy(data_dict):
        if 'anchor_locs' not in data_dict:
            data_dict['anchor_locs'] = torch.zeros(3)
        if 'anchor_orientation' not in data_dict:
            data_dict['anchor_orientation'] = torch.zeros(4)
            data_dict['anchor_orientation'][-1] = 1   # xyzw
        if 'img_fts' not in data_dict:
            data_dict['img_fts'] = torch.zeros(3, 224, 224)   # currently hardcode to 224x224
        if 'img_masks' not in data_dict:
            data_dict['img_masks'] = torch.LongTensor([0]).bool()

        for key in LEOMIX_REQUIRED_KEYS:
            if key not in data_dict:
                raise ValueError(f"Key {key} is missing in LeoMix data_dict")
        return data_dict

    def load_rscan(self, scan_id):
        scan_path = os.path.join(self.rscan_base, '3RScan-ours-align', scan_id)
        pcd_data = torch.load(os.path.join(scan_path, 'pcd-align.pth'))
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        inst_to_label = torch.load(os.path.join(scan_path, 'inst_to_label.pth'))

        # build obj_pcds
        obj_pcds = {}
        for inst_id in inst_to_label.keys():
            mask = instance_labels == inst_id
            obj_pcds.update({inst_id: pcds[mask]})

        return {'obj_pcds': obj_pcds}

    def load_scannet(self, scan_id):
        scan = {}
        pcd_data = torch.load(os.path.join(self.scannet_base, 'scan_data',
                                           'pcd_with_global_alignment', f'{scan_id}.pth'))
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)

        obj_pcds = {}
        for i in range(instance_labels.max() + 1):
            mask = instance_labels == i
            obj_pcds.update({i: pcds[mask]})

        scan['obj_pcds'] = obj_pcds

        if hasattr(self, 'pc_type') and self.pc_type == 'pred':
            # Mask3D proposals
            mask_path = os.path.join(self.scannet_base, 'mask', f'{str(scan_id)}.mask.npz')
            obj_masks = np.array(sparse.load_npz(mask_path).todense())[:50, :]
            obj_pcds_pred = []
            for i in range(obj_masks.shape[0]):
                mask = obj_masks[i]
                obj_pcds_pred.append(pcds[mask == 1, :])
            scan['obj_pcds_pred'] = obj_pcds_pred

        return scan

    def preprocess_pcd(self, obj_pcds, return_anchor=False, rot_aug=True):
        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug=rot_aug)

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        for i, obj_pcd in enumerate(obj_pcds):
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())

            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            if return_anchor and i == 0:
                # Select a loc within the obj bbox as the anchor.
                anchor_loc = obj_pcd[:, :3].min(0) + np.random.rand(3) * obj_size

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)
            obj_pcd = obj_pcd[pcd_idxs]

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.sqrt((obj_pcd[:, :3]**2).sum(1)).max()
            if max_dist < 1e-6:   # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0)).float()
        obj_locs = torch.from_numpy(np.array(obj_locs)).float()
        if return_anchor:
            anchor_loc = torch.from_numpy(anchor_loc).float()
        else:
            anchor_loc = torch.zeros(3)

        return obj_fts, obj_locs, anchor_loc

    def _split_sentence(self, sentence, max_length, prefix=''):
        # only split during training
        if self.split == 'train' and len(prefix + sentence) > max_length:
            all_caps = []
            sents = sentence.split('. ')
            tmp = prefix
            for i in range(len(sents)):
                if len(tmp + sents[i] + '. ') > max_length:
                    all_caps.append(tmp)
                    tmp = prefix
                tmp += sents[i] + '. '

            all_caps.append(tmp)   # last chunk

            # final check
            ret = []
            for cap in all_caps:
                if len(cap) <= max_length:
                    ret.append(cap)
            return ret
        else:
            return [prefix + sentence]


# alignment

@DATASET_REGISTRY.register()
class LeoCap3D(LeoBase):
    situation_pool = Leo_situation_pool
    instruction_pool = Leo_objcap_instruction_pool

    def __init__(self, cfg, split='train'):
        super().__init__()
        self.split = split
        self.cap3d_root = cfg.data.cap3d.cap3d_root
        self.num_points = cfg.data.cap3d.num_points

        logger.info(f"Loading LeoCap3D {split}-set language")
        self.create_obj_cap_dict(self.cap3d_root)
        if split == 'train':
            self.obj_ids = self.obj_ids[:-1000]
        else:
            self.obj_ids = self.obj_ids[-1000:]
        logger.info(f"Finish loading LeoCap3D {split}-set language, collected {len(self.obj_ids)} data")

    def create_obj_cap_dict(self, cap3d_root):
        obj_csv = pd.read_csv(os.path.join(cap3d_root, 'Cap3D_automated_Objaverse_no3Dword.csv'), header=None)
        self.obj_ids = []
        self.obj_cap_dict = {}
        for obj_id, cap in zip(obj_csv[0].values, obj_csv[1].values):
            # remove redundant quotation marks, here we do not directly strip because the mark may appear only at one side
            if cap.startswith('"') and cap.endswith('"'):
                cap = cap.strip('"')
            elif cap.startswith("'") and cap.endswith("'"):
                cap = cap.strip("'")

            self.obj_ids.append(obj_id)
            self.obj_cap_dict[obj_id] = cap

    def load_obj_pcd(self, obj_id):
        pcd = torch.load(
            os.path.join(self.cap3d_root, f'Cap3D_pcs_pt/{obj_id}.pt'),
            map_location='cpu'
        )   # (6, 16384)
        pcd = rearrange(pcd, 'c n -> n c')   # (16384, 6), xyz (m) + rgb (uint8)
        pcd[:, 3:] = pcd[:, 3:] / 127.5 - 1   # (16384, 6), xyz (m) + rgb (float, [-1, 1])
        return pcd

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        obj_id = self.obj_ids[index]
        obj_pcd = self.load_obj_pcd(obj_id)
        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd([obj_pcd.numpy()], return_anchor=True)

        data_dict = self.get_prompts(
            instruction=random.choice(self.instruction_pool),
            situation=random.choice(self.situation_pool),
        )
        data_dict.update({
            'source': 'objaverse',
            'scene_id': obj_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': self.obj_cap_dict[obj_id],
        })
        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class LeoObjSceneCap(LeoBase):
    situation_pool = Leo_situation_pool
    instruction_pool = Leo_objcap_instruction_pool

    def __init__(self, cfg, split='train'):
        super().__init__()
        assert split == 'train', "LeoObjSceneCap only supports training during the alignment stage"
        self.split = split
        self.rscan_base = cfg.data.obj_scene_cap.rscan_base
        self.scannet_base = cfg.data.obj_scene_cap.scannet_base
        self.num_points = cfg.data.obj_scene_cap.num_points
        self.max_obj_len = cfg.data.obj_scene_cap.max_obj_len
        self.max_caption_length = int(cfg.llm.max_out_len / LLAMA_TOKEN_SENT_RATIO)

        logger.info("Loading LeoObjSceneCap train-set language")
        self.scan_ids, self.lang_data = self.load_anno(cfg.data.obj_scene_cap.anno_dir)
        # scan_ids may be repeatitive
        logger.info(f"Finish loading LeoObjSceneCap train-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {'3rscan': {}, 'scannet': {}}

    def load_anno(self, anno_dir):
        # may contain both 3RScan and ScanNet
        scan_ids = []
        scan_caps = []
        for fname in os.listdir(anno_dir):
            if '3rscan' in fname.lower():
                with open(os.path.join(anno_dir, fname)) as f:
                    json_data = json.load(f)
                if 'scanscribe' in fname.lower():
                    for meta_anno in json_data:
                        cap = meta_anno['sentence']
                        all_caps = self._split_sentence(
                            sentence='. '.join(cap.split('. ')[1:]),
                            max_length=self.max_caption_length,
                            prefix=cap.split('. ')[0] + '. ',
                        )
                        for c in all_caps:
                            scan_ids.append({
                                'source': '3rscan',
                                'scan_id': meta_anno['scan_id'],
                            })
                            scan_caps.append({
                                'obj_id': meta_anno['object_id'],
                                'caption': c,
                            })
                else:
                    # 3rscan_prompted
                    for k, v in json_data.items():
                        for obj_str, obj_v in v.items():
                            obj_id = int(obj_str.split('-')[-1])
                            for meta_anno in obj_v:
                                cap = meta_anno['response']
                                all_caps = self._split_sentence(
                                    sentence='. '.join(cap.split('. ')[1:]),
                                    max_length=self.max_caption_length,
                                    prefix=cap.split('. ')[0] + '. ',
                                )
                                for c in all_caps:
                                    scan_ids.append({
                                        'source': '3rscan',
                                        'scan_id': k,
                                    })
                                    scan_caps.append({
                                        'obj_id': obj_id,
                                        'caption': c,
                                    })
            elif 'scannet' in fname.lower():
                # referit3d
                with jsonlines.open(os.path.join(anno_dir, fname), 'r') as f:
                    for item in f:
                        cap = item['utterance']
                        all_caps = self._split_sentence(
                            sentence='. '.join(cap.split('. ')[1:]),
                            max_length=self.max_caption_length,
                            prefix=cap.split('. ')[0] + '. ',
                        )
                        for c in all_caps:
                            scan_ids.append({
                                'source': 'scannet',
                                'scan_id': item['scan_id'],
                            })
                            scan_caps.append({
                                'obj_id': item['target_id'],
                                'caption': c,
                            })

        return scan_ids, scan_caps

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_meta = self.scan_ids[index]
        scan_source = scan_meta['source']
        scan_id = scan_meta['scan_id']

        lang_meta = self.lang_data[index]
        obj_id = lang_meta['obj_id']
        obj_caption = lang_meta['caption']

        # load pcds
        if scan_id not in self.scan_data[scan_source]:
            if scan_source == '3rscan':
                self.scan_data['3rscan'][scan_id] = self.load_rscan(scan_id)
            elif scan_source == 'scannet':
                self.scan_data['scannet'][scan_id] = self.load_scannet(scan_id)
        obj_pcds = self.scan_data[scan_source][scan_id]['obj_pcds'].copy()
        # Dict: { int: np.ndarray (N, 6) }

        selected_obj_pcds = [ obj_pcds[obj_id] ]
        remained_obj_idx = [i for i in obj_pcds.keys() if i != obj_id]
        if self.split == 'train':
            random.shuffle(remained_obj_idx)
        selected_obj_pcds.extend([obj_pcds[i] for i in remained_obj_idx[: self.max_obj_len - 1]])

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=True)
        data_dict = self.get_prompts(
            instruction=random.choice(self.instruction_pool),
            situation=random.choice(self.situation_pool),
        )
        data_dict.update({
            'source': scan_source,
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': obj_caption,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class LeoSceneCap(LeoBase):
    instruction_pool = Leo_scenecap_instruction_pool

    def __init__(self, cfg, split='train'):
        super().__init__()
        self.split = split
        self.rscan_base = cfg.data.scene_cap.rscan_base
        self.num_points = cfg.data.scene_cap.num_points
        self.max_obj_len = cfg.data.scene_cap.max_obj_len
        self.max_caption_length = int(cfg.llm.max_out_len / LLAMA_TOKEN_SENT_RATIO)

        logger.info(f"Loading LeoSceneCap {split}-set language")
        self.scan_ids, self.lang_data, self.scan_insts = self.load_anno(cfg.data.scene_cap.anno_dir)
        # scan_ids may be repeatitive
        if self.split == 'train':
            self.scan_ids = self.scan_ids[:-500]
            self.lang_data = self.lang_data[:-500]
            self.scan_insts = self.scan_insts[:-500]
        else:
            self.scan_ids = self.scan_ids[-500:]
            self.lang_data = self.lang_data[-500:]
            self.scan_insts = self.scan_insts[-500:]
        logger.info(f"Finish loading LeoSceneCap {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        scan_ids = []
        scan_caps = []
        scan_insts = []   # relevant instances
        for fname in os.listdir(anno_dir):
            if fname.endswith('json'):
                with open(os.path.join(anno_dir, fname)) as f:
                    json_data = json.load(f)
                for k, v in json_data.items():
                    for meta_anno in v:
                        scene_graph = eval(meta_anno['query'])
                        insts = [int(s.split('-')[-1]) for s in scene_graph.keys()]

                        cap = meta_anno['response']
                        all_caps = self._split_sentence(
                            sentence='. '.join(cap.split('. ')[1:]),
                            max_length=self.max_caption_length,
                            prefix=cap.split('. ')[0] + '. ',
                        )
                        for c in all_caps:
                            scan_caps.append(c)
                            scan_ids.append(k)
                            scan_insts.append(insts)

        return scan_ids, scan_caps, scan_insts

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        caption = self.lang_data[index]
        scan_insts = self.scan_insts[index]

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_rscan(scan_id)
        obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()
        # Dict: { int: np.ndarray (N, 6) }

        selected_obj_pcds = [obj_pcds[i] for i in scan_insts]
        num_selected_objs = len(selected_obj_pcds)
        if num_selected_objs >= self.max_obj_len:
            if self.split == 'train':
                random.shuffle(selected_obj_pcds)
            selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]
        else:
            # select from remaining objs
            remained_obj_idx = [i for i in obj_pcds.keys() if i not in scan_insts]
            if self.split == 'train':
                random.shuffle(selected_obj_pcds)
                random.shuffle(remained_obj_idx)
            selected_obj_pcds.extend(
                [obj_pcds[i] for i in remained_obj_idx[: self.max_obj_len - num_selected_objs]]
            )

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)
        data_dict = self.get_prompts(
            instruction=random.choice(self.instruction_pool),
            situation="",
        )
        data_dict.update({
            'source': '3rscan',
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': caption,
        })

        return self.check_output_and_fill_dummy(data_dict)


# instruction tuning

@DATASET_REGISTRY.register()
class LeoScan2Cap(LeoBase):
    situation_pool = Leo_situation_pool
    instruction_pool = Leo_objcap_instruction_pool

    def __init__(self, cfg, split):
        super().__init__()
        self.scannet_base = cfg.data.scan2cap.scannet_base
        self.num_points = cfg.data.scan2cap.num_points
        self.max_obj_len = cfg.data.scan2cap.max_obj_len
        self.max_caption_length = int(cfg.llm.max_out_len / LLAMA_TOKEN_SENT_RATIO)

        if split == 'train':
            self.split = 'train'
            self.pc_type = 'gt'
        else:
            self.split = 'val'   # test set inaccessible
            self.pc_type = getattr(cfg.data.scan2cap, 'pc_type', 'gt')        

        self.iou_threshold = getattr(cfg.data.scan2cap, 'iou_thres', 0.5)

        logger.info(f"Loading LeoScan2Cap {split}-set language")
        self.scan_ids, self.lang_data, self.corpus_cache = self.load_anno(cfg.data.scan2cap.anno_dir)
        # scan_ids may be repeatitive
        logger.info(f"Finish loading LeoScan2Cap {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        anno_file = os.path.join(anno_dir, 'scannet_scanrefer.jsonl')
        split_file = os.path.join(anno_dir, f'scannetv2_{self.split}.txt')
        split_scan_ids = set([x.strip() for x in open(split_file, 'r')])
        scan_ids = []
        scan_caps = []
        corpus_cache = []
        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if item['scan_id'] in split_scan_ids:
                    scan_id = item['scan_id']
                    obj_id = int(item['target_id'])
                    obj_name = item['instance_type']
                    key = f'{scan_id}|{obj_id}|{obj_name}'
                    if self.split != 'train' and key in corpus_cache:
                        continue
                    # only evaluate once per obj instance
                    corpus_cache.append(key)
                    cap = item['utterance']
                    all_caps = self._split_sentence(
                        sentence='. '.join(cap.split('. ')[1:]),
                        max_length=self.max_caption_length,
                        prefix=cap.split('. ')[0] + '. ',
                    )
                    for c in all_caps:
                        scan_ids.append(item['scan_id'])
                        scan_caps.append({
                            'obj_id': item['target_id'],
                            'caption': c,
                        })

        return scan_ids, scan_caps, corpus_cache

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        lang_meta = self.lang_data[index]
        obj_id = lang_meta['obj_id']
        obj_caption = lang_meta['caption']
        corpus_key = self.corpus_cache[index]

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_scannet(scan_id)

        if self.pc_type == 'gt':
            iou_flag = 1
            obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()   # Dict{int: np.ndarray (N, 6)}
            selected_obj_pcds = [ obj_pcds[obj_id] ]
            remained_obj_idx = [i for i in obj_pcds.keys() if i != obj_id]
        else:
            # only for evaluation with object proposals
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred'].copy()   # List[np.ndarray (N, 6)]
            gt_pcd = self.scan_data[scan_id]['obj_pcds'][obj_id].copy()
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_obj_id_pred = -1
            overlap_obj_id_list = []
            max_iou = self.iou_threshold
            iou_flag = 0
             # find max iou
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                current_iou = eval_ref_one_sample(
                    construct_bbox_corners(obj_center, obj_box_size),
                    construct_bbox_corners(gt_center, gt_box_size)
                )
                if current_iou >= max_iou:
                    iou_flag = 1
                    tgt_obj_id_pred = i
                    max_iou = current_iou
                if current_iou >= 0.25:
                    # this list includes tgt_obj_id_pred, as long as iou_thres >= 0.25
                    overlap_obj_id_list.append(i)
            selected_obj_pcds = [ obj_pcds[tgt_obj_id_pred] ]
            selected_obj_pcds.extend([obj_pcds[i] for i in overlap_obj_id_list if i != tgt_obj_id_pred])
            remained_obj_idx = [i for i in range(len(obj_pcds)) if i not in overlap_obj_id_list]

        num_selected_obj = len(selected_obj_pcds)
        if num_selected_obj >= self.max_obj_len:
            selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]
        else:
            if self.split == 'train':
                random.shuffle(remained_obj_idx)
            selected_obj_pcds.extend(
                [obj_pcds[i] for i in remained_obj_idx[: self.max_obj_len - num_selected_obj]]
            )

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=True)
        data_dict = self.get_prompts(
            instruction=random.choice(self.instruction_pool),
            situation=random.choice(self.situation_pool),
        )
        data_dict.update({
            'source': 'scannet',
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': obj_caption,
            'iou_flag': torch.LongTensor([iou_flag]).bool(),
            'corpus_key': corpus_key,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class LeoScanQA(LeoBase):
    def __init__(self, cfg, split):
        super().__init__()
        self.scannet_base = cfg.data.scanqa.scannet_base
        self.num_points = cfg.data.scanqa.num_points
        self.max_obj_len = cfg.data.scanqa.max_obj_len

        if split == 'train':
            self.split = 'train'
            self.pc_type = 'gt'
        else:
            self.split = 'val'   # test split inaccessible
            self.pc_type = getattr(cfg.data.scanqa, 'pc_type', 'gt')

        logger.info(f"Loading LeoScanQA {split}-set language")
        self.scan_ids, self.lang_data, self.scan_insts = self.load_anno(cfg.data.scanqa.anno_dir)
        # scan_ids may be repeatitive
        logger.info(f"Finish loading LeoScanQA {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        scan_ids = []
        scan_qa_pairs = []
        scan_insts = []
        anno_file = os.path.join(anno_dir, f'ScanQA_v1.0_{self.split}.json')
        with open(anno_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for item in json_data:
            scan_ids.append(item['scene_id'])
            scan_qa_pairs.append({
                'q': item['question'],   # str
                'a': [s.strip() for s in item['answers']],   # list of str
            })
            # try to parse concerned objects
            insts = item['object_ids']
            scan_insts.append(insts)

        return scan_ids, scan_qa_pairs, scan_insts

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        qa_dict = self.lang_data[index]
        scan_insts = self.scan_insts[index]
        question = qa_dict['q']   # str
        answer_list = qa_dict['a']   # list of str

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_scannet(scan_id)

        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()   # Dict{int: np.ndarray (N, 6)}
            selected_obj_pcds = [ obj_pcds[obj_id] for obj_id in scan_insts ]
            remained_obj_idx = [i for i in obj_pcds.keys() if i not in scan_insts]
        else:
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred'].copy()   # List[np.ndarray (N, 6)]
            gt_center = []
            gt_box_size = []
            for obj_id in scan_insts:
                gt_pcd = self.scan_data[scan_id]['obj_pcds'][obj_id].copy()
                center, box_size = convert_pc_to_box(gt_pcd)
                gt_center.append(center)
                gt_box_size.append(box_size)

            # select proposals with high IoU with question-relevant gt pcds
            selected_obj_pcds = []
            remained_obj_idx = []
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                proposal_selected = False
                for center, box_size in zip(gt_center, gt_box_size):
                    if eval_ref_one_sample(
                        construct_bbox_corners(obj_center, obj_box_size),
                        construct_bbox_corners(center, box_size)
                    ) >= 0.25:
                        selected_obj_pcds.append(obj_pcds[i])
                        proposal_selected = True
                        break
                if not proposal_selected:
                    remained_obj_idx.append(i)

        num_selected_objs = len(selected_obj_pcds)
        if num_selected_objs >= self.max_obj_len:
            if self.split == 'train':
                random.shuffle(selected_obj_pcds)
            selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]
        else:
            if self.split == 'train':
                random.shuffle(selected_obj_pcds)
                random.shuffle(remained_obj_idx)
            selected_obj_pcds.extend(
                [obj_pcds[i] for i in remained_obj_idx[: self.max_obj_len - num_selected_objs]]
            )

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)

        data_dict = self.get_prompts(
            instruction=question,
            situation="",
        )
        data_dict.update({
            'source': 'scannet',
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': random.choice(answer_list) if self.split == 'train' else answer_list,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class LeoSQA3D(LeoBase):
    situation_pool = Leo_situation_pool

    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        self.scannet_base = cfg.data.sqa3d.scannet_base
        self.num_points = cfg.data.sqa3d.num_points
        self.max_obj_len = cfg.data.sqa3d.max_obj_len
        if split == 'train':
            self.pc_type = 'gt'
        else:
            self.pc_type = getattr(cfg.data.sqa3d, 'pc_type', 'gt')
        
        logger.info(f"Loading LeoSQA3D {split}-set language")
        self.scan_ids, self.lang_data = self.load_anno(cfg.data.sqa3d.anno_dir)
        # scan_ids may be repeatitive
        logger.info(f"Finish loading LeoSQA3D {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        scan_ids = []
        sqa_annos = []

        question_file = os.path.join(anno_dir, f'v1_balanced_questions_{self.split}_scannetv2.json')
        with open(question_file, 'r', encoding='utf-8') as f:
            question_data = json.load(f)['questions']
        question_map = {}
        for item in question_data:
            question_map[item['question_id']] = {
                's': [item['situation']] + item['alternative_situation'],   # list of str
                'q': item['question'],   # str
            }

        anno_file = os.path.join(anno_dir, f'v1_balanced_sqa_annotations_{self.split}_scannetv2.json')
        with open(anno_file, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)['annotations']
        for item in anno_data:
            scan_ids.append(item['scene_id'])
            sqa_annos.append({
                's': question_map[item['question_id']]['s'],   # list of str
                'q': question_map[item['question_id']]['q'],   # str
                'a': [meta['answer'] for meta in item['answers']],   # list of str
                'pos': np.array(list(item['position'].values())),   # array (3,)
                'rot': np.array(list(item['rotation'].values())),   # array (4,)
            })

        return scan_ids, sqa_annos

    def __len__(self):
        return len(self.scan_ids)

    def convert_person_view(self, sentence):
        # first-person view to second-person view
        forms = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'am': 'are'}
        def translate(word):
            if word.lower() in forms:
                return forms[word.lower()]
            return word
        result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(sentence)])
        return result.capitalize()

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        sqa_dict = self.lang_data[index]
        situation = sqa_dict['s']   # list of str
        question = sqa_dict['q']   # str
        answer_list = sqa_dict['a']   # list of str
        pos = sqa_dict['pos']   # array, (3,)
        rot = sqa_dict['rot']   # array, (4,)

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_scannet(scan_id)

        # sqa3d has no annotations of question-relevant objs
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()   # Dict{int: np.ndarray (N, 6)}
            obj_pcds = list(obj_pcds.values())   # to list
        else:
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred'].copy()   # List[np.ndarray (N, 6)]

        if self.split == 'train':
            random.shuffle(obj_pcds)
        selected_obj_pcds = obj_pcds[:self.max_obj_len]

        obj_fts, obj_locs, _ = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)

        if self.split == 'train':
            # augmentation for train
            situation = random.choice(situation)
        else:
            # fix for eval
            situation = situation[0]

        question_type = get_sqa_question_type(question)

        data_dict = self.get_prompts(
            instruction=self.convert_person_view(question),
            situation=random.choice(self.situation_pool) + ' ' + self.convert_person_view(situation),
        )
        data_dict.update({
            'source': 'scannet',
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'situation': situation,
            'anchor_locs': torch.from_numpy(pos).float(),
            'anchor_orientation': torch.from_numpy(rot).float(),
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': random.choice(answer_list) if self.split == 'train' else answer_list,
            'sqa_type': question_type,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class Leo3RScanQA(LeoBase):
    """json format
    {
        "1776ad80-4db7-2333-8b18-f02ef42f3569": {
            "query": "{'floor-1': {'relations': [], 'attribute': {'material': 'wooden', 'shape': 'flat', 'color': 'brown'}},}",
            "response": [
                {
                    "Q": "What is the material of the floor?",
                    "T": "floor-1",
                    "A": ["wooden"]
                },
                {
                    "Q": "What color are the walls?",
                    "T": "wall-2, wall-3",
                    "A": ["white"]
                },
            ]
        },
    }
    """

    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        self.rscan_base = cfg.data.rscan_qa.rscan_base
        self.num_points = cfg.data.rscan_qa.num_points
        self.max_obj_len = cfg.data.rscan_qa.max_obj_len

        logger.info(f"Loading Leo3RScanQA {split}-set language")
        self.scan_ids, self.lang_data, self.scan_insts = self.load_anno(cfg.data.rscan_qa.anno_dir)
        # scan_ids may be repeatitive
        if self.split == 'train':
            self.scan_ids = self.scan_ids[:-4000]
            self.lang_data = self.lang_data[:-4000]
            self.scan_insts = self.scan_insts[:-4000]
        else:
            self.scan_ids = self.scan_ids[-4000:]
            self.lang_data = self.lang_data[-4000:]
            self.scan_insts = self.scan_insts[-4000:]
        logger.info(f"Finish loading Leo3RScanQA {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        scan_ids = []
        scan_qa_pairs = []
        scan_insts = []
        for fname in os.listdir(anno_dir):
            if fname.endswith('json'):
                with open(os.path.join(anno_dir, fname)) as f:
                    json_data = json.load(f)
                for k, v in json_data.items():
                    for meta_anno in v['response']:
                        # try to parse concerned objects
                        try:
                            insts = meta_anno['T'].split(', ')
                            insts = [int(s.split('-')[-1]) for s in insts]
                        except:
                            insts = []
                        scan_insts.append(insts)
                        scan_ids.append(k)
                        scan_qa_pairs.append({
                            'q': meta_anno['Q'],   # str
                            'a': [a.strip() for a in meta_anno['A']],   # list of str
                        })

        return scan_ids, scan_qa_pairs, scan_insts

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        qa_dict = self.lang_data[index]
        scan_insts = self.scan_insts[index]
        question = qa_dict['q']
        answer_list = qa_dict['a']

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_rscan(scan_id)
        obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()
        # Dict: { int: np.ndarray (N, 6) }

        # crop objects to max_obj_len, select relevant objs first
        selected_obj_pcds = [obj_pcds[i] for i in scan_insts]

        num_selected_objs = len(selected_obj_pcds)
        if num_selected_objs >= self.max_obj_len:
            if self.split == 'train':
                random.shuffle(selected_obj_pcds)
            selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]
        else:
            # select from remaining objs
            remained_obj_idx = [i for i in obj_pcds.keys() if i not in scan_insts]
            if self.split == 'train':
                random.shuffle(selected_obj_pcds)
                random.shuffle(remained_obj_idx)
            for i in remained_obj_idx[: self.max_obj_len - num_selected_objs]:
                selected_obj_pcds.append(obj_pcds[i])

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)
        data_dict = self.get_prompts(
            instruction=question,
            situation="",
        )
        data_dict.update({
            'source': '3rscan',
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': random.choice(answer_list) if self.split == 'train' else answer_list,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class Leo3RScanPlan(LeoBase):
    instruction_prefix_pool = Leo_plan_instruction_pool
    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        self.rscan_base = cfg.data.rscan_plan.rscan_base
        self.num_points = cfg.data.rscan_plan.num_points
        self.max_obj_len = cfg.data.rscan_plan.max_obj_len

        logger.info(f"Loading Leo3RScanPlan {split}-set language")
        self.scan_ids, self.lang_data = self.load_anno(cfg.data.rscan_plan.anno_dir)
        # scan_ids may be repeatitive
        if self.split == 'train':
            self.scan_ids = self.scan_ids[:-500]
            self.lang_data = self.lang_data[:-500]
        else:
            self.scan_ids = self.scan_ids[-500:]
            self.lang_data = self.lang_data[-500:]
        logger.info(f"Finish loading Leo3RScanPlan {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        scan_ids = []
        lang_data = []
        for fname in os.listdir(anno_dir):
            if fname.endswith('json'):
                with open(os.path.join(anno_dir, fname)) as f:
                    json_data = json.load(f)
                for k, v in json_data.items():
                    for meta_anno in v['response']:
                        scan_ids.append(k)
                        lang_data.append({
                            'goal': meta_anno['instruction'],
                            'plan': meta_anno['plan'],
                        })
                        # no split operation as we assume the response length has been processed in advance

        return scan_ids, lang_data

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        goal_plan_pair = self.lang_data[index]
        goal = goal_plan_pair['goal']
        plan = goal_plan_pair['plan']

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_rscan(scan_id)
        obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()
        # Dict: { int: np.ndarray (N, 6) }

        selected_obj_pcds = list(obj_pcds.values())
        if self.split == 'train':
            random.shuffle(selected_obj_pcds)
        selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)
        data_dict = self.get_prompts(
            instruction=random.choice(self.instruction_prefix_pool) + ': ' + goal.lower(),
            situation="",
        )
        data_dict.update({
            'source': '3rscan',
            'scece_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': plan,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class Leo3RScanDialog(LeoBase):
    r"""The format of json file
    {
        'scan_id': {
            'query': scene graph,
            'response': dialogues, # list of list, [dialog_1, dialog_2, ...]
        }
    }
    The format of dialog_i
    [
        {'role': 'Human', 'content': 'What is the color of the sofa?'},
        {'role': 'Robot', 'content': 'The color of the sofa is red. '},
        {'role': 'Human', 'content': 'Is the sofa in good condition?'},
        {'role': 'Robot', 'content': 'No, the sofa is in an old state. '},
    ]
    Dialogue for Vicuna: "USER: Who are you? ASSISTANT: I am Vicuna.</s>USER: What can you do? ASSISTANT:"
    """
    def __init__(self, cfg, split):
        super().__init__()
        self.split = split
        self.rscan_base = cfg.data.rscan_dialog.rscan_base
        self.num_points = cfg.data.rscan_dialog.num_points
        self.max_obj_len = cfg.data.rscan_dialog.max_obj_len

        logger.info(f"Loading Leo3RScanDialog {split}-set language")
        self.scan_ids, self.lang_data = self.load_anno(cfg.data.rscan_dialog.anno_dir)
        # scan_ids may be repeatitive
        if self.split == 'train':
            self.scan_ids = self.scan_ids[:-500]
            self.lang_data = self.lang_data[:-500]
        else:
            self.scan_ids = self.scan_ids[-500:]
            self.lang_data = self.lang_data[-500:]
        logger.info(f"Finish loading Leo3RScanDialog {split}-set language, collected {len(self.scan_ids)} data")

        self.scan_data = {}

    def load_anno(self, anno_dir):
        scan_ids = []
        lang_data = []
        for fname in os.listdir(anno_dir):
            if fname.endswith('json'):
                with open(os.path.join(anno_dir, fname)) as f:
                    json_data = json.load(f)
                for k, v in json_data.items():
                    dialogs = v['response']
                    for dialog in dialogs:
                        assert dialog[0]['role'] == 'Human', "Dialogue should start with Human"
                        assert len(dialog) > 1, "Dialogue should contain Robot responses"
                        history = f"USER: {dialog[0]['content']} ASSISTANT:"
                        scan_ids.append(k)
                        lang_data.append({
                            'history': history,
                            'response': dialog[1]['content'].strip(),
                        })
                        for i in range(1, len(dialog)):
                            meta_anno = dialog[i]
                            if i % 2 == 0 and i+1 < len(dialog):
                                # Human
                                history += f"USER: {meta_anno['content']} ASSISTANT:"
                                scan_ids.append(k)
                                lang_data.append({
                                    'history': history,
                                    'response': dialog[i+1]['content'].strip(),
                                })
                            else:
                                # Robot
                                history += f" {meta_anno['content'].strip()}</s>"

        return scan_ids, lang_data

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, index):
        scan_id = self.scan_ids[index]
        dialog_pair = self.lang_data[index]
        history = dialog_pair['history']
        response = dialog_pair['response']

        # load pcds
        if scan_id not in self.scan_data:
            self.scan_data[scan_id] = self.load_rscan(scan_id)
        obj_pcds = self.scan_data[scan_id]['obj_pcds'].copy()
        # Dict: { int: np.ndarray (N, 6) }

        selected_obj_pcds = list(obj_pcds.values())
        if self.split == 'train':
            random.shuffle(selected_obj_pcds)
        selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)
        data_dict = self.get_prompts(
            instruction=None,
            dialogue=history,
            situation="",
        )
        data_dict.update({
            'source': '3rscan',
            'scene_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_locs': anchor_loc,
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'output_gt': response,
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class LeoMix(Dataset):
    mapping = {
        'cap3d': LeoCap3D,
        'obj_scene_cap': LeoObjSceneCap,
        'scene_cap': LeoSceneCap,
        'scan2cap': LeoScan2Cap,
        'scanqa': LeoScanQA,
        'sqa3d': LeoSQA3D,
        'rscan_qa': Leo3RScanQA,
        'rscan_plan': Leo3RScanPlan,
        'rscan_dialog': Leo3RScanDialog,
    }

    def __init__(self, cfg, split):
        self.datasets = []
        self.ratio = cfg.task.leomix.ratio
        logger.info(f"LeoMix about to load: {cfg.task.leomix.mix}")
        for dataset in cfg.task.leomix.mix:
            self.datasets.append(self.mapping[dataset](cfg, split))

        if type(self.ratio) == int or type(self.ratio) == float:
            self.index_range = list(np.cumsum([int(len(d)*self.ratio) for d in self.datasets]))
        else:
            self.index_range = list(np.cumsum([int(len(d)*self.ratio[i]) for i, d in enumerate(self.datasets)]))
        self.index_range = [0] + self.index_range
        logger.info(f"Indices of LeoMix datasets: {self.index_range}")

    def __len__(self):
        return self.index_range[-1]

    @staticmethod
    def streamline_output(data_dict):
        new_data_dict = {}
        for key in LEOMIX_REQUIRED_KEYS:
            if key not in data_dict:
                raise ValueError(f"Key {key} is missing in LeoMix data_dict")
            else:
                new_data_dict[key] = data_dict[key]
        return new_data_dict

    def __getitem__(self, index):
        for i in range(len(self.index_range)-1):
            if self.index_range[i] <= index < self.index_range[i+1]:
                data_dict = self.datasets[i][index-self.index_range[i]]
                break

        return self.streamline_output(data_dict)
