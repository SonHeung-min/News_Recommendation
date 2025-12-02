import inspect

import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer


def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    val_metrics_list: list[RecMetrics] = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        # Inference
        batch["news_histories"] = batch["news_histories"].to(device)
        batch["candidate_news"] = batch["candidate_news"].to(device)
        batch["target"] = batch["target"].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()

        # Calculate Metrics
        val_metrics_list.append(RecEvaluator.evaluate_all(y_true, y_score))

    rec_metrics = RecMetrics(
        **{
            "ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
    )

    return rec_metrics



def train(
    pretrained: str,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_len: int,
    resume_from_checkpoint: str | None = None, # <--- THÊM THAM SỐ NÀY
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    
    # ... (Các bước 0, 1, 2, 3: Định nghĩa tham số, Khởi tạo Model, Load Data & Create Dataset) ...

    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)
    
    # Chỉ tạo thư mục mới nếu KHÔNG tiếp tục từ checkpoint
    if resume_from_checkpoint is None:
        model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Nếu tiếp tục, Trainer sẽ tự động tìm các file trong thư mục này
        model_save_dir = Path(resume_from_checkpoint).parent 

    """
    1. Init Model
    """
    logging.info("Initialize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )

    """
    2. Load Data & Create Dataset
    """
    logging.info("Initialize Dataset")

    train_news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    train_behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
    # Lưu ý: Nếu bạn gặp lỗi OOM (Out-of-Memory), bạn có thể cần bỏ device=device khỏi MINDTrainDataset
    train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device) 

    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logging.info("Training Start")
    training_args_params = {
        "output_dir": model_save_dir,
        "logging_strategy": "steps",
        "save_total_limit": 5,
        "lr_scheduler_type": "constant",
        "weight_decay": weight_decay,
        "optim": "adamw_torch",
        "save_strategy": "epoch",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": epochs,
        "remove_unused_columns": False,
        "logging_dir": LOG_OUTPUT_DIR,
        "logging_steps": 1,
        "report_to": "none",
    }
    # Check which evaluation strategy parameter name is available
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_args_params["eval_strategy"] = "no"
    elif "evaluation_strategy" in sig.parameters:
        training_args_params["evaluation_strategy"] = "no"

    training_args = TrainingArguments(**training_args_params)

    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # GỌI HÀM TRAIN VỚI THAM SỐ RESUME_FROM_CHECKPOINT
    trainer.train(resume_from_checkpoint=resume_from_checkpoint) # <--- SỬ DỤNG THAM SỐ RESUME

    """
    4. Evaluate model by Validation Dataset
    """
    logging.info("Evaluation")
    metrics = evaluate(trainer.model, eval_dataset, device)
    logging.info(metrics.dict())


# Cần import Path và Union nếu Python < 3.10
from pathlib import Path
from typing import Union 

@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        
        # --- BƯỚC MỚI: Xử lý resume training ---
        
        # Đặt đường dẫn đến thư mục checkpoint (ví dụ: 'MODEL_OUTPUT_DIR/timestamp/checkpoint-614')
        # Dùng Path để đảm bảo tương thích đa nền tảng
        # Thay thế đường dẫn placeholder này bằng đường dẫn thực tế của bạn
        checkpoint_path = "/kaggle/working/News_Recommendation/outputs/model/2025-11-28/09-06-09/checkpoint-614"  # <--- ĐƯỜNG DẪN CHECKPOINT CẦN TIẾP TỤC TRAIN
        
        if checkpoint_path and Path(checkpoint_path).is_dir():
            resume_path: Union[str, None] = checkpoint_path
            logging.info(f"Resuming training from checkpoint: {resume_path}")
        else:
            resume_path = None
            logging.info("Starting new training session.")
        
        # --- Gọi hàm train đã cập nhật ---
        train(
            cfg.pretrained,
            cfg.npratio,
            cfg.history_size,
            cfg.batch_size,
            cfg.gradient_accumulation_steps,
            cfg.epochs,
            cfg.learning_rate,
            cfg.weight_decay,
            cfg.max_len,
            resume_from_checkpoint=resume_path, # <--- TRUYỀN ĐƯỜNG DẪN CHECKPOINT VÀO
        )
        
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    main()