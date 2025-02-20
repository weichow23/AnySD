import numpy as np
import PIL
import torch
from huggingface_hub import hf_hub_download

task_embs_book = {
    # global
    "background_change": 1,
    "tone_transfer": 2,
    "style_change": 3,
    # viewpoint
    "movement": 4,
    "resize": 5,
    "rotation_change": 6,
    "outpaint": 7,
    # Visual-related tasks
    "visual_bbox": 8,
    "visual_depth": 9,
    "visual_material_transfer": 10,
    "visual_reference": 11,
    "visual_scribble": 12,
    "visual_segment": 13,
    "visual_sketch": 14
}

global_embs_book = {
    "background_change": 0,
    "tone_transfer": 1,
    "style_change": 2,
}
viewpoint_embs_book = {
    "movement": 0,
    "resize": 1,
    "outpaint": 2,
    "rotation_change": 3,
}

def visual_embs_book(expert_name):
    for key in task_embs_book.keys():
        if key.startswith(expert_name):
            return {key: 0}
    raise ValueError(f"Expert name {expert_name} not found in task_embs_book.")

def choose_book(expert_name):
    if expert_name == 'viewpoint':
        expert_num = 4
        task_book = viewpoint_embs_book
    elif expert_name == 'global':
        expert_num = 3
        task_book = global_embs_book
    elif expert_name == 'all':
        expert_num = 14
        task_book = task_embs_book
    else:
        expert_num = 1
        task_book = visual_embs_book(expert_name)
    return expert_num, task_book

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def convert_to_np(image, resolution):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def get_experts_dir(repo_id = "WeiChow/AnySD"):
    filename = "README.md" 
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return local_path.replace(filename, "experts/")

def print_format_tree(obj, indent=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{indent}{key}:")
            print_format_tree(value, indent + "  ")
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}Tensor of shape {list(obj.shape)}")
    else:
        print(f"{indent}{type(obj)}")

if __name__ == "__main__":
    ckpt = "../ip_adapter_plus/ip_adapter_sd15.bin"
    sd = torch.load(ckpt, map_location="cpu")
    print_format_tree(sd)
    