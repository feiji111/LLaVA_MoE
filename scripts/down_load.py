import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="liuhaotian/LLaVA-Pretrain",
    repo_type="dataset",
    max_workers=8
)