# Environment 1
# AnySD: keep the same environment with the AnySD model part
# -------------------------------------------------------------------------------------------
# Environment 2
conda create --name anybench python=3.10 -y
conda activate anybench
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers # pip install .
pip install git+https://github.com/openai/CLIP.git
pip install transformers==4.45.2
# -------------------------------------------------------------------------------------------
# Environment 3
# Ultra + Unicond: ultra need a different diffusers
conda create --name ultra python=3.10 -y
conda activate ultra
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install sentencepiece protobuf
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
cd diffusers && pip install -e . # download the https://github.com/HaozheZhao/UltraEdit
pip install git+https://github.com/openai/CLIP.git
pip install transformers==4.24.0

# -------------------------------------------------------------------------------------------
# Model Checkpoints Download
cd anybench/checkpoints/hive
gcloud storage cp gs://sfr-hive-data-research/checkpoints/hive_rw_condition.ckpt hive_rw_condition.ckpt
gcloud storage cp gs://sfr-hive-data-research/checkpoints/hive_rw.ckpt hive_rw.ckpt
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

cd ../unicond  # notice that unicond need pip install transformers==4.24.0, so we use utral base
huggingface-cli download lllyasviel/ControlNet --include annotator/ckpts/ --local-dir . --local-dir-use-symlinks False
pip install open_clip_torch opencv-python pytorch_lightning==2.4.0 scikit-image
