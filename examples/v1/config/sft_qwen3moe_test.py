from transformers import AutoProcessor, AutoTokenizer
from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3VLMoE30BA3Config
from xtuner.v1.train import TrainerConfig

model_path = "/vast/projects/jacobrg/agentic-science/yimengz/hf_cache/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_cfg = Qwen3VLMoE30BA3Config()  # fake tokenizer vocab size for tiny model
# processor = AutoProcessor.from_pretrained(model_path)

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
        "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=model_path),
    },
    {
        "dataset": DatasetConfig(
            name="media",
            anno_path="tests/resource/mllm_sft_single_image_example_data.jsonl",
            media_root="tests/",
            sample_ratio=2.0,
            class_name="VLMJsonlDataset",
        ),
        "tokenize_fn": Qwen3VLTokenizeFnConfig(processor_path=model_path),
    },
] * 10
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config, pack_max_length=pack_max_length, collator="qwen3_vl_sft_collator"
)

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-4, foreach=False)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0)

# trainer config
trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    load_from=model_path,
    tokenizer_path=model_path,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=32,
    intra_layer_micro_batch=4,   # this is the per device micro batch size in xtuner
    total_epoch=1,
    work_dir="work_dir/run_2_sft_qwen3moe_test",
)
