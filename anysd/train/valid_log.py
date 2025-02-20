import wandb
import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
from contextlib import nullcontext
import requests
from torchvision.transforms import transforms
import clip
import torch
import PIL
from anybench.eval.utils import eval_distance, eval_clip_i, eval_clip_t, is_all_black

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

def download_image(url):
   if not os.path.exists(url):
      path = requests.get(url, stream=True).raw
   else:
      path = url
   image = PIL.Image.open(path)
   image = PIL.ImageOps.exif_transpose(image)
   image = image.convert("RGB").resize((512, 512))
   return image

@torch.inference_mode()
def stagei_edit(pipe, image_urls, prompts):
    if type(image_urls) == list:
        assert type(prompts) == list
        if isinstance(image_urls[0], str):
            image = [PIL.Image.open(i_url).convert("RGB").resize((512, 512)) for i_url in image_urls]
        else:
            image = [i_url.convert("RGB").resize((512, 512)) for i_url in image_urls]
    else:
        if isinstance(image_urls, str):
            image = PIL.Image.open(image_urls).convert("RGB").resize((512, 512))
        else:
            image = image_urls.convert("RGB").resize((512, 512))

    edited_image = pipe(prompts,
         image=image,
         num_inference_steps=64,
         image_guidance_scale=3,
         guidance_scale=3,
         generator=torch.Generator("cuda").manual_seed(0),
         ).images

    return edited_image # List[PIL.Image]

def log_validation_stagei(
    pipeline,
    args,
    accelerator,
    generator,
    logger
):
    logger.info(f"Running validation...")
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        ds = load_dataset("facebook/emu_edit_test_set_generations")
        dataset = ds['test'].select(range(50))
        batch_size = 100
        image_pairs, image_pairs_dir = [], []
        wandb_list = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            images = stagei_edit(pipe=pipeline.to(accelerator.device), image_urls=[item['image'] for item in batch],
                                 prompts=[item['instruction'] for item in batch])
            for item, img in zip(batch, images):
                if not is_all_black(img):  # skip black
                    image_pairs.append([item['edited_image'], img, item['output_caption']])
                    image_pairs_dir.append(
                        [item['edited_image'], img,
                         item['output_caption'], item['image'], item['input_caption']])
                    wandb_list.append({'item': item, 'img': img})
            model, transform = clip.load("ViT-B/32")
            clip_i = eval_clip_i(image_pairs=image_pairs, model=model, transform=transform, url_flag=False)
            clip_t, _ = eval_clip_t(image_pairs=image_pairs, model=model, transform=transform, url_flag=False)
            l1 = eval_distance(image_pairs=image_pairs, metric='l1', url_flag=False)
            l2 = eval_distance(image_pairs=image_pairs, metric='l2', url_flag=False)

            dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            dino_model.eval()
            dino_model.to("cuda")
            dino_transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dino_score = eval_clip_i(image_pairs=image_pairs, model=dino_model, transform=dino_transform,
                                     url_flag=False, metric='dino')
            print(f"CLIP-I: {clip_i:.3f}  CLIP-T: {clip_t:.3f}  "
                  f"DINO: {dino_score:.3f}  L1: {l1:.3f}  L2: {l2:.3f}")

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for item in wandb_list[:5]:
                wandb_table.add_data(wandb.Image(item['item']['image']), wandb.Image(item['img']), item['item']['instruction'])

            tracker.log({
                "validation": wandb_table,
                "clip_i": clip_i,
                "clip_t": clip_t,
                "dino_score": dino_score,
                "l1": l1,
                "l2": l2
            })

def log_validation_stageii(
    pipeline,
    args,
    accelerator,
    logger,
    task_book
):
    logger.info(f"Running validation...")
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        test_tasks = list(task_book.keys())

        if args.expert_name in ['viewpoint', 'global']:  # more than 3 will lead to async error in my GPU
            test_tasks = test_tasks[-2:]
        elif args.expert_name == 'local':
            test_tasks = random.sample(test_tasks, 2)

        for dim in test_tasks:
            batch_size = 1 # only support bs=1
            image_pairs = []
            wandb_list = []

            with open(f"./anybench/dataset/anybench/{dim}/edit_result.json") as fp:
                data_json = json.load(fp)

            # for i in tqdm(range(0, len(data_json), batch_size)):
            #     batch = data_json[i:i + batch_size]
            for item in tqdm(data_json):
                img = pipeline(
                    prompt=item['edit'],
                    reference_image=download_image(f"./anybench/dataset/{item['visual_input']}") if ('visual_input' in item.keys() and item['visual_input'] is not None)  else None,
                    original_image=download_image(f"./anybench/dataset/{item['image_file']}"),
                    guidance_scale=1.5,
                    num_inference_steps=100,
                    original_image_guidance_scale=2,
                    reference_image_guidance_scale=0.8,
                    adapter_name="expert-only",
                    e_code=task_book[dim],
                )[0]

                if not is_all_black(img):  # skip black
                    image_pairs.append([download_image(f"./anybench/dataset/{item['edited_file']}"), img, item["output"] if 'output' in item.keys() else None])
                    wandb_list.append({'item': item, 'img': img})

            model, transform = clip.load("ViT-B/32")
            clip_i = eval_clip_i(image_pairs=image_pairs, model=model, transform=transform, url_flag=False)
        
            if image_pairs[0][2] is not None:
                clip_t, _ = eval_clip_t(image_pairs=image_pairs, model=model, transform=transform, url_flag=False)
            else:
                clip_t = 0

            l1 = eval_distance(image_pairs=image_pairs, metric='l1', url_flag=False)

            dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            dino_model.eval()
            dino_model.to("cuda")
            dino_transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dino_score = eval_clip_i(image_pairs=image_pairs, model=dino_model, transform=dino_transform,
                                    url_flag=False, metric='dino')
            print(f"CLIP-I: {clip_i:.3f}  CLIP-T: {clip_t:.3f}  DINO: {dino_score:.3f}  L1: {l1:.3f}")

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    for item in wandb_list[:5]:
                        wandb_table.add_data(wandb.Image(download_image(f"./anybench/dataset/{item['item']['image_file']}")), wandb.Image(item['img']), item['item']['edit'])

                    tracker.log({
                        f"{dim}-validation": wandb_table,
                        f"{dim}-clip_i": clip_i,
                        f"{dim}-clip_t": clip_t,
                        f"{dim}-dino_score": dino_score,
                        f"{dim}-l1": l1,
                    })