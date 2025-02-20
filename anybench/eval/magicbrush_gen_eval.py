'''
CUDA_VISIBLE_DEVICES=4 PYTHONPATH='./' python anybench/eval/magicbrush_gen_eval.py
'''
from __future__ import annotations
import os.path
from tqdm import tqdm
import os
from anybench.eval.utils import eval_magicbrush_score, load_all
from anybench.models.method import UniEditModel

def main(input_path, output_path, skip_iter, model_name):
    data_json = load_all(input_path, only_instr=False)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = UniEditModel(name=model_name)

    # iterative edit
    if skip_iter:
        print("Skip Iterative (Mutli-Turn) Editing......")
    else:
        print("Iterative (Mutli-Turn) Editing......")
        # for idx, datas in enumerate(data_json):
        for image_id, datas in tqdm(data_json.items()):
            for turn_id, data in enumerate(datas):
                image_name = data['input']
                image_dir = image_name.split('-')[0]
                if turn_id == 0:  # first enter
                    image_path = os.path.join(input_path, image_dir, image_name)
                else:
                    image_path = save_output_img_path

                save_output_dir_path = os.path.join(output_path, image_dir)
                if not os.path.exists(save_output_dir_path):
                    os.makedirs(save_output_dir_path)
                if turn_id == 0:
                    save_output_img_path = os.path.join(save_output_dir_path, image_dir+'_1.png')
                else:
                    save_output_img_path = os.path.join(save_output_dir_path, image_dir+'_iter_' +str(turn_id + 1)+'.png')
                
                if os.path.exists(save_output_img_path):
                    print('already generated, skip')
                    continue
                print('image_name', image_name)
                print('image_path', image_path)
                print('save_output_img_path', save_output_img_path)
                print('edit_instruction', data['instruction'])
                model.magic_edit(image_path=image_path,
                                 save_output_img_path=save_output_img_path,
                                 data=data)

    print("Independent (Single-Turn) Editing......")
    # for idx, datas in enumerate(data_json):
    for image_id, datas in tqdm(data_json.items()):
        for turn_id, data in enumerate(datas):
            image_name = data['input']
            image_dir = image_name.split('-')[0]
            image_path = os.path.join(input_path, image_dir, image_name)

            save_outut_dir_path = os.path.join(output_path, image_dir)
            if not os.path.exists(save_outut_dir_path):
                os.makedirs(save_outut_dir_path)
            if turn_id == 0:
                save_output_img_path = os.path.join(save_outut_dir_path, image_dir+'_1.png')
                # if os.path.exists(save_output_img_path):  # already generated in iterative editing
                #     print('already generated in iterative editing')
                #     continue
            else:
                save_output_img_path = os.path.join(save_outut_dir_path, image_dir+'_inde_' +str(turn_id + 1)+'.png')
            if os.path.exists(save_output_img_path):  # already generated in iterative (multi-turn) editing.
                print('already generated, skip')
                continue
            print('image_name', image_name)
            print('image_path', image_path)
            print('save_output_img_path', save_output_img_path)
            print('edit_instruction', data['instruction'])

            model.magic_edit(image_path=image_path,
                             save_output_img_path=save_output_img_path,
                             data=data)

if __name__ == "__main__":
    model_name = 'anysd'
    input_path = './anybench/dataset/magicbrush/test/images'
    save_path = f"./anybench/results/magicbrush/{model_name}/"
    skip_iter = True

    main(input_path, save_path, skip_iter, model_name) # gen

    eval_magicbrush_score(generated_path = save_path,
                          gt_path=input_path,
                          caption_path = './anybench/dataset/magicbrush/test/local_descriptions.json',
                          skip_iter =skip_iter,
                          metric=['clip-d'],  # 'l1','clip-i','dino','clip-t'
                          clip_model_type="ViT-L/14",
                          dino_model_type='dino_vitb16'
                          )