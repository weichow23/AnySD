import os
from tqdm import tqdm
from anysd.src.model import AnySDPipeline, choose_expert
from anysd.train.valid_log import download_image
from anysd.src.utils import choose_book, get_experts_dir

if __name__ == "__main__":
    expert_file_path = get_experts_dir(repo_id="WeiChow/AnySD")
    book_dim, book = choose_book('all')
    task_embs_checkpoints = expert_file_path + "task_embs.bin"
    adapter_checkpoints = {
        "global": expert_file_path + "global.bin",
        "viewpoint": expert_file_path + "viewpoint.bin",
        "visual_bbox": expert_file_path + "visual_bbox.bin",
        "visual_depth": expert_file_path + "visual_dep.bin",
        "visual_material_transfer": expert_file_path + "visual_mat.bin",
        "visual_reference": expert_file_path + "visual_ref.bin",
        "visual_scribble": expert_file_path + "visual_scr.bin",
        "visual_segment": expert_file_path + "visual_seg.bin",
        "visual_sketch": expert_file_path + "visual_ske.bin",
    }

    pipeline = AnySDPipeline(adapters_list=adapter_checkpoints, task_embs_checkpoints=task_embs_checkpoints)

    os.makedirs('./assets/anysd-test/', exist_ok=True)
    case = [
        {
            "edit": "Put on a pair of sunglasses",
            "edit_type": 'general',
            "image_file": "./assets/woman.jpg"
        },
        {
            "edit": "Make her a wizard",
            "edit_type": 'general',
            "image_file": "./assets/woman.jpg"
        },
        {
            "edit": "Make it evening",
            "edit_type": 'general',
            "image_file": "./assets/bear.jpg"
        },
        {
            "edit": "Make it underwater",
            "edit_type": 'general',
            "image_file": "./assets/bear.jpg"
        },
        {
            "edit": "Change the season to winter",
            "edit_type": 'general',
            "image_file": "./assets/car.jpg"
        },
        {
            "edit": "Change the color of the car to yellow",
            "edit_type": 'general',
            "image_file": "./assets/car.jpg"
        },
        {
            "edit": 'replace the cat to a [V*]',
            "edit_type": 'visual_reference',
            "refence_image_file": "./assets/bewitchingcat.jpg",
            "image_file": "./assets/window.jpg"
        }
    ]

    for index, item in enumerate(tqdm(case)):
        mode = choose_expert(mode=item["edit_type"])
        if mode == 'general':
            images = pipeline(
                prompt=item['edit'],
                original_image=download_image(item['image_file']),
                guidance_scale=3,
                num_inference_steps=100,
                original_image_guidance_scale=3,
                adapter_name="general",
            )[0]
        else:
            images = pipeline(
                prompt=item['edit'],
                reference_image=download_image(item['refence_image_file']) if ('refence_image_file' in item.keys() and item['refence_image_file'] is not None)  else None,
                original_image=download_image(item['image_file']),
                guidance_scale=1.5,
                num_inference_steps=100,
                original_image_guidance_scale=2,
                reference_image_guidance_scale=0.8,
                adapter_name=mode,
                e_code=book[item["edit_type"]],
            )[0]

        images.save(f"./assets/anysd-test/{index}.jpg")
