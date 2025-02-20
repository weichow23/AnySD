import random
import torch
import numpy as np
from torchvision import transforms
from transformers import CLIPImageProcessor
from datasets import load_dataset, concatenate_datasets
import yaml
from anysd.src.utils import convert_to_np, tokenize_captions
import contextlib

expert_name_list = ["visual_ref", "visual_mat", "visual_dep", "visual_bbox", "visual_seg", "visual_ske",
                    "visual_scr", "viewpoint",
                    "local", "global", "implicit"]
local_expert_prefix = ['replace', 'add', 'remove', 'color_alter', 'appearance_alter', "material_change", 'action_change',
                  'counting','textual_change']
global_expert_prefix = ['tune_transfer', 'background_change', 'style_change']
viewpoint_expert_prefix = ['movement', 'outpaint', 'resize', 'rotation_change']
implicit_expert_prefix = ['relation']
visual_ref_expert_prefix = ['visual_material_transfer', 'visual_reference']
visual_cond_expert_prefix = ['visual_material_transfer', 'visual_reference']

def can_convert_to_float(s):
    try:
        float(s)
        return True
    except:
        return False

class AnyEditMixtureDatasetStageI(torch.utils.data.Dataset):
    # zw@zju: support SD 1.5
    def __init__(self, args, tokenizers, accelerator):
        self.args = args
        self.tokenizers=tokenizers
        self.accelerator=accelerator
        self.train_transforms = transforms.Compose(
            [
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            ]
        )

        self.examples = self._get_examples()

    def _hf_dataset_sample(self, ds, sample_ratio):
        num_samples = len(ds)

        if sample_ratio == 1 or sample_ratio == 1.0:
            return ds
        elif sample_ratio < 1:
            k_samples = int(num_samples * sample_ratio)
            return ds.shuffle(seed=self.args.seed).select(range(k_samples))
        else:
            repeat_count = int(sample_ratio)
            extended_dataset = concatenate_datasets([ds] * repeat_count)
            remaining_samples = int(num_samples * (sample_ratio - repeat_count))
            extended_dataset = extended_dataset.shuffle(seed=self.args.seed).select(range(remaining_samples))  # 追加剩余的样本

            return extended_dataset

    def _get_examples(self):
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        print('[INFO] Begin loading dataset')
        random.seed(self.args.seed)  # Ensure reproducibility

        if self.accelerator is not None:
            context_manager = self.accelerator.main_process_first()
        else:
            # Dummy context manager when accelerator is not used
            context_manager = contextlib.nullcontext()

        with context_manager:
            with open(self.args.yaml_file, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)

            ds_combined = None
            for key_1, value_1 in yaml_data.items():  # key1: repo name
                ds_repo = load_dataset(key_1)
                sampled_ds_total = None
                for key_2, value_2 in value_1.items():        # key2: split name
                    ds_split = ds_repo[key_2.replace('_', '')]  # _ should not in split name 
                    if can_convert_to_float(value_2): # only float sample ratio
                        sampled_ds = self._hf_dataset_sample(ds=ds_split, sample_ratio=float(value_2))
                        if sampled_ds_total is None:
                                sampled_ds_total = sampled_ds
                        else:
                            sampled_ds_total = concatenate_datasets([sampled_ds_total, sampled_ds])  

                if ds_combined is None:
                    ds_combined = sampled_ds_total
                else:
                    ds_combined = concatenate_datasets([ds_combined, sampled_ds_total])

            if hasattr(self.args, 'max_train_samples') and self.args.max_train_samples is not None:
                ds_combined.shuffle(seed=self.args.seed).select(range(self.args.max_train_samples))

        if hasattr(self.args, 'dataset_repeat_num') and self.args.dataset_repeat_num is not None:
            ds_combined = concatenate_datasets([ds_combined] * self.args.dataset_repeat_num)

        print('[INFO] End loading dataset')
        return ds_combined

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        original_image = convert_to_np(item["image_file"], self.args.resolution)
        edited_image = convert_to_np(item["edited_file"], self.args.resolution)
        images = np.concatenate([original_image, edited_image])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        images = self.train_transforms(images)

        original_image_transformed, edited_image_transformed = images.chunk(2)
        original_image_transformed = original_image_transformed.reshape(
            3, self.args.resolution, self.args.resolution
        )
        edited_image_transformed = edited_image_transformed.reshape(
            3, self.args.resolution, self.args.resolution
        )
        prompt_embeds = tokenize_captions(captions=[item["edit_instruction"]], tokenizer=self.tokenizers)

        return {
            "original_pixel_values": original_image_transformed,
            "edited_pixel_values": edited_image_transformed,
            "input_ids": prompt_embeds[0]
        }

def collate_fn_anyedit(examples):
    original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    if "reference_clip_image" in examples[0].keys():
        # stage II
        if examples[0]["reference_clip_image"] == torch.zeros([1, 3, 224, 224]):
            reference_clip_images = torch.cat([example["reference_clip_image"] for example in examples], dim=0)
        else:
            reference_clip_images = torch.cat([CLIPImageProcessor()(example["reference_clip_image"]) for example in examples])
        
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "reference_clip_images":reference_clip_images,
            "input_ids": input_ids,
            "edit_code": [example["edit_code"] for example in examples]  # codebook number
        }
    else:
        # stage I
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }


class AnyEditMixtureDatasetStageII(AnyEditMixtureDatasetStageI):
    def __init__(self, args, tokenizers, accelerator, task_book):
        super().__init__(args=args, tokenizers=tokenizers, accelerator=accelerator)
        self.clip_image_processor = CLIPImageProcessor()
        self.task_embs_book = task_book

    def __getitem__(self, idx):
        item = self.examples[idx]

        original_image = convert_to_np(item["image_file"], self.args.resolution)
        edited_image = convert_to_np(item["edited_file"], self.args.resolution)
        images = np.concatenate([original_image, edited_image])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        images = self.train_transforms(images)

        original_image_transformed, edited_image_transformed = images.chunk(2)
        original_image_transformed = original_image_transformed.reshape(
            3, self.args.resolution, self.args.resolution
        )
        edited_image_transformed = edited_image_transformed.reshape(
            3, self.args.resolution, self.args.resolution
        )
        prompt_embeds = tokenize_captions(captions=[item["edit_instruction"]], tokenizer=self.tokenizers)

        if "visual_input" in item.keys() and item["visual_input"] is not None:
            # can't use clip_image_processor in dataloader, that will cause too much time
            # reference_clip_image = self.clip_image_processor(images=item["visual_input"], return_tensors="pt").pixel_values
            reference_clip_image = item["visual_input"]
        else:
            reference_clip_image = torch.zeros([1, 3, 224, 224])

        if "edit_type" in item.keys() and item["edit_type"] is not None:
            edit_type = item["edit_type"]
            if edit_type == "visual_material_transfer":
                edit_type = "material_change"
        else:
            edit_type = None

        return {
            "original_pixel_values": original_image_transformed,
            "edited_pixel_values": edited_image_transformed,
            "reference_clip_image": reference_clip_image,
            "input_ids": prompt_embeds[0],
            "edit_code": self.task_embs_book[edit_type]
        }