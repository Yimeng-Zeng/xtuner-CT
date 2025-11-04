from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import InternS1MiniConfig
from xtuner.v1.train import TrainerConfig


# model config
model_cfg = InternS1MiniConfig()

# dataset and dataloader config
sample_max_length = 4096
pack_max_length = 4096

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="pure_text",
            anno_path="tests/resource/mllm_sft_text_example_data.jsonl",
            sample_ratio=1.0,
            class_name="VLMJsonlDataset",
        ),
        "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(
            name="media",
            anno_path="tests/resource/mllm_sft_single_image_example_data.jsonl",
            media_root="tests/",
            sample_ratio=2.0,
            class_name="VLMJsonlDataset",
        ),
        "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config, pack_max_length=pack_max_length, collator="intern_s1_vl_sft_collator"
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-4, foreach=False)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0)

# trainer config
trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    load_from="/vast/home/y/yimengz/vast_j_partition/hf_cache/hub/models--internlm--Intern-S1-mini/snapshots/d790aca537fe624b309620e64ed9c50f88011c8c/",
    tokenizer_path="/vast/home/y/yimengz/vast_j_partition/hf_cache/hub/models--internlm--Intern-S1-mini/snapshots/d790aca537fe624b309620e64ed9c50f88011c8c/",
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=32,
    total_epoch=1,
    work_dir="work_dir/run_1_sft_intern_s1_mini_test",
)
