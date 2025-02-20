import os
import logging
from tqdm import tqdm
from anybench.models.method import UniEditModel
import torch
from PIL import Image
from torchvision.transforms import transforms
from datasets import load_dataset
import clip
from anybench.eval.utils import eval_distance, eval_clip_i, eval_clip_t, is_all_black

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_dataset(bench_name):
    if bench_name == 'emu_test':
        # val: 3589  test: 2022, no mask in data
        ds = load_dataset("facebook/emu_edit_test_set_generations")
        return ds['test']
    elif bench_name == 'emu_val':
        ds = load_dataset("facebook/emu_edit_test_set_generations")
        return ds['validation']
    else:
        raise ValueError(f"Unknown benchmark named {bench_name}")

def eval_emu(dataset, save_path):
    image_pairs = []
    for item in tqdm(dataset):
        if not is_all_black(f"{save_path}/{item['idx']}.png"):  # skip the black image
            image_pairs.append(
                [item['edited_image'], Image.open(f"{save_path}/{item['idx']}.png").convert('RGB'),
                 item['output_caption']])

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

    print(f" CLIP-I: {clip_i:.3f}  CLIP-T: {clip_t:.3f}  DINO: {dino_score:.3f}  L1: {l1:.3f}  L2: {l2:.3f}")
    logging.info(f" CLIP-I: {clip_i:.3f}  CLIP-T: {clip_t:.3f}  DINO: {dino_score:.3f}  L1: {l1:.3f}  L2: {l2:.3f}")

if __name__ == '__main__':
    bench_name = "emu_val" # choices=['emu_test', 'emu_val']
    dataset = get_dataset(bench_name)
    # dataset = dataset.select(range(200))

    model_name = 'anysd'
    save_path = f"./anybench/results/{bench_name}/{model_name}/"
    os.makedirs(save_path, exist_ok=True)

    log_file = f"./anybench/results/{bench_name}/log.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)

    batch_size = 128

    model = UniEditModel(name=model_name)
    logging.info(f"{model_name}")

    if model_name in ['nulltext', 'nulltext-1.5', 'nulltext-xl']:
        for batch in tqdm(dataset):
            model.emu_edit(batch=batch, bench_name=bench_name, save_path=save_path)
    else:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            model.emu_edit(batch=batch, bench_name=bench_name, save_path=save_path)

    eval_emu(dataset=dataset, save_path=save_path)