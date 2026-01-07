# Thêm các import cần thiết nếu chưa có
import inspect
import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from safetensors.torch import load_file as safe_load_file
from pathlib import Path

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

# --- Giữ lại hàm evaluate và các hàm khác từ code gốc của bạn ---

# ... (Hàm evaluate) ...
def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    # ... (Nội dung hàm evaluate) ...
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    list_raw_scores = []
    val_metrics_list: list[RecMetrics] = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        # Inference
        batch["news_histories"] = batch["news_histories"].to(device)
        batch["candidate_news"] = batch["candidate_news"].to(device)
        batch["target"] = batch["target"].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        # Save raw scores for each sample
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy().tolist()[:-1]
        list_raw_scores.append(y_score)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()[:-1]
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()[:-1]

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

    return rec_metrics, list_raw_scores



# --- Hàm mới để tải mô hình đã lưu và đánh giá trên tập validation ---

# Thêm thư viện safetensors vào imports ở đầu file nếu chưa có
from safetensors.torch import load_file as safe_load_file 

# ... (các imports khác) ...

def load_model_for_test(
    model_path_dir: str | Path, # Đổi tên tham số để nhấn mạnh đây là ĐƯỜNG DẪN THƯ MỤC
    pretrained: str,
    history_size: int,
    max_len: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> RecMetrics:
    """
    Tải mô hình NRMS đã lưu (từ file .safetensors) và đánh giá trên tập validation.

    Args:
        model_path_dir: Đường dẫn đến THƯ MỤC checkpoint đã lưu (ví dụ: 'output/.../checkpoint-XXXX').
        pretrained: Tên/đường dẫn của Pretrained Language Model (PLM) đã dùng.
        history_size: Kích thước lịch sử đọc tin tức tối đa.
        max_len: Chiều dài token tối đa cho mỗi tin tức.
        device: Thiết bị để chạy inference.
    
    Returns:
        RecMetrics: Các metrics đánh giá.
    """
    logging.info(f"Start loading model from directory: {model_path_dir}")
    model_path_dir = Path(model_path_dir)
    
    """
    1. Definite Parameters & Init Model Architecture
    """
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    
    # Khởi tạo lại kiến trúc mô hình
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )
    
    """
    2. Load Model Weights
    """
    # Xây dựng đường dẫn file .safetensors
    # Trainer mặc định lưu trọng số dưới tên này.
    weights_path = model_path_dir / "model.safetensors" 
    
    if not weights_path.exists():
        # Thêm kiểm tra tương thích với pytorch_model.bin cũ
        weights_path_pt = model_path_dir / "pytorch_model.bin"
        if weights_path_pt.exists():
             weights_path = weights_path_pt
        else:
             raise FileNotFoundError(f"Không tìm thấy file trọng số model.safetensors hoặc pytorch_model.bin tại thư mục: {model_path_dir}")
        
    logging.info(f"Loading weights from: {weights_path}")
    
    if weights_path.suffix == ".safetensors":
        # Tải bằng safetensors
        state_dict = safe_load_file(weights_path, device="cpu") # Tải về CPU trước để tránh lỗi OOM
    else:
        # Tải bằng torch.load cho .bin hoặc .pt
        state_dict = torch.load(weights_path, map_location=device)
        
    # Chuyển state_dict sang thiết bị đích (nếu nó chưa ở đó)
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    
    nrms_net.load_state_dict(state_dict)
    logging.info("Model weights loaded successfully.")
    
    """
    3. Load Data & Create Dataset
    """
    # ... (giữ nguyên phần này) ...
    logging.info("Initialize Dataset")
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)
    
    """
    4. Evaluate model by Validation Dataset
    """ 
    logging.info("Evaluation Start")
    metrics, list_raw_scores = evaluate(nrms_net, eval_dataset, device)
    logging.info(f"Evaluation Metrics: {metrics.dict()}")
    
    with open("/kaggle/working/scores.txt", "w") as f:
        for idx, row in enumerate(list_raw_scores, start=1):
            f.write(f"{idx} {row}\n")

    
    

# --- Cập nhật hàm main để gọi hàm test ---

@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        
        # Giả định bạn đã hoàn thành bước train và có một checkpoint/output_dir để test
        # Bạn cần thay thế model_to_test_path bằng đường dẫn thực tế đến checkpoint
        # Ví dụ: path đến thư mục MODEL_OUTPUT_DIR/timestamp/checkpoint-XXXX
        # Ví dụ: model_to_test_path = 'MODEL_OUTPUT_DIR/2025-12-01_22-00-00/checkpoint-300'
        
        # Nếu bạn muốn train trước rồi mới test
        # train(
        #     cfg.pretrained,
        #     cfg.npratio,
        #     cfg.history_size,
        #     cfg.batch_size,
        #     cfg.gradient_accumulation_steps,
        #     cfg.epochs,
        #     cfg.learning_rate,
        #     cfg.weight_decay,
        #     cfg.max_len,
        # )
        
        # --- Khối code mới để test mô hình đã lưu ---
        
        # Đặt đường dẫn đến checkpoint mà bạn muốn test
        # Ví dụ: đường dẫn này thường là thư mục cuối cùng Trainer lưu (checkpoint-...)
        # Bạn có thể thêm tham số vào TrainConfig nếu muốn chỉ định nó qua config
        model_to_test_path = "/kaggle/input/checkpoint-1842/pytorch/default/1"
        
        if model_to_test_path:
            logging.info("Starting model testing on validation set.")
            load_model_for_test(
                model_to_test_path,
                cfg.pretrained,
                cfg.history_size,
                cfg.max_len,
            )
        else:
             logging.info("No model path provided. Exiting.")
        
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    main()