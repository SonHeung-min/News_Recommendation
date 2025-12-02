from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "distilbert-base-uncased"
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 8  # batch_size = 16 x 8 = 128
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_len: int = 30


@dataclass
class EvalConfig:
    random_seed: int = 42
    pretrained: str = "distilbert-base-uncased"
    history_size: int = 50
    max_len: int = 30
    model_path: str = (
        "output/model/2025-11-28/09-06-09/checkpoint-614"  # Path to model checkpoint, if empty will use most recent model in MODEL_OUTPUT_DIR
    )


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)
cs.store(name="eval_config", node=EvalConfig)
