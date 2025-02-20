'''
This is a sample script to show how anybench can be used:

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH='./' python3 anybench/eval/anybench_gen_eval.py
'''
import os
import json
from tqdm import tqdm
import PIL
import torch
from torchvision.transforms import transforms
import clip
import logging
from anybench.eval.utils import eval_distance, eval_clip_i, eval_clip_t, is_all_black
from anybench.models.method import UniEditModel

local_dim_list = ['replace', 'add', 'remove', 'color_alter', 'appearance_alter', "material_change", 'action_change',
                  'counting','textual_change']
global_dim_list = ['tone_transfer', 'background_change', 'style_change']
camera_dim_list = ['movement', 'outpaint', 'resize', 'rotation_change']
implicit_dim_list = ['relation', 'implicit_change']
test_dim_list = local_dim_list + global_dim_list + camera_dim_list + implicit_dim_list
visual_dim_list = ['visual_material_transfer', 'visual_reference', 'visual_bbox', 'visual_depth', 'visual_scribble',
                   'visual_segment', 'visual_sketch']
dim_list = test_dim_list + visual_dim_list

if __name__ == '__main__':
    model_name = "anydm"
    mode = 'e'  # 'g' for only geneartion, 'e' for only evaluation

    if mode == 'e':
        model = UniEditModel(name='anysd')

    for dim in implicit_dim_list:
        assert model_name != "unicond" or dim in visual_dim_list, 'Unicond only support visual prompt editing'

        logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')
        save_path = f"./anybench/results/anybench/{dim}/{model_name}/"
        os.makedirs(save_path, exist_ok=True)

        with open(f"./anybench/dataset/anybench/{dim}/edit_result.json") as fp:
            data_json = json.load(fp)

        image_pairs = []
        for item in tqdm(data_json):
            if mode != 'e':
                model.anybench_edit(
                    dim=dim, item=item, logging=logging,
                    save_path=f"results/{model_name}/{dim}/{item['image_file'].split('/')[-1]}",
                    image_url=f"your path/anyedit/dataset/{item['image_file']}",
                    cond_image_url=f"your path/anyedit/dataset/{item['visual_input']}",
                )

            input_image_path = f"anybench/dataset/{item['edited_file']}"
            output_image_path = f"anybench/results/{model_name}/{dim}-1500/{item['image_file'].split('/')[-1]}"

            if mode != 'g':
                if not is_all_black(output_image_path):
                    image_pairs.append(
                    [PIL.Image.open(input_image_path),  # gt
                     PIL.Image.open(output_image_path).convert('RGB'),  # output
                     item['output'] if 'output' in item.keys() else None])

        # ----------------- evaluation ref to eval.py
        if mode != 'g':
            if len(image_pairs) == 0:
                pass
                logging.info(f"Model {model_name} skip {dim}")
            else:
                logging.info(f"Model {model_name} dim {dim}")

                clip_model, transform = clip.load("ViT-B/32")
                clip_i = eval_clip_i(image_pairs=image_pairs, model=clip_model, transform=transform, url_flag=False)
                if 'output' in data_json[0].keys() and data_json[0]['output'] is not None:
                    clip_t, _ = eval_clip_t(image_pairs=image_pairs, model=clip_model, transform=transform, url_flag=False)
                else:
                    clip_t = -100
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

                print(f" CLIP-I: {clip_i:.3f}  CLIP-T: {clip_t:.3f}  DINO: {dino_score:.3f}  L1: {l1:.3f}")
                logging.info(f" CLIP-I: {clip_i:.3f}  CLIP-T: {clip_t:.3f}  DINO: {dino_score:.3f}  L1: {l1:.3f}")