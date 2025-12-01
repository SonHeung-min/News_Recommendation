import pathlib

import hydra
import numpy as np
import safetensors.torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import ModelOutput

from config.config import EvalConfig
from const.path import MIND_SMALL_TEST_DATASET_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDValDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from utils.logger import logging
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer


def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    val_metrics_list: list[RecMetrics] = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDTestDataset"):
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


def load_pretrained_model(
    model_path: str | pathlib.Path,
    pretrained: str,
    history_size: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> NRMS:
    """
    Load pretrained NRMS model from checkpoint.

    Parameters
    ----------
    model_path : str | pathlib.Path
        Path to the model checkpoint directory or state_dict file
    pretrained : str
        Pretrained model name (e.g., "distilbert-base-uncased")
    history_size : int
        History size used during training
    device : torch.device
        Device to load model on

    Returns
    -------
    NRMS
        Loaded NRMS model
    """
    logging.info(f"Loading model from {model_path}")

    # Initialize model architecture
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )

    # Load weights
    model_path = pathlib.Path(model_path)
    if model_path.is_dir():
        # If it's a directory, try to load from checkpoint
        # Check for pytorch_model.bin or model.safetensors
        checkpoint_path = None

        # First, check if there are checkpoint subdirectories (checkpoint-0, checkpoint-1, etc.)
        checkpoint_dirs = sorted(
            [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else -1,
            reverse=True,
        )

        if checkpoint_dirs:
            # Use the latest checkpoint directory
            for checkpoint_dir in checkpoint_dirs:
                if (checkpoint_dir / "pytorch_model.bin").exists():
                    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
                    break
                elif (checkpoint_dir / "model.safetensors").exists():
                    checkpoint_path = checkpoint_dir / "model.safetensors"
                    break

        # If no checkpoint subdirectory found, check root directory
        if checkpoint_path is None:
            if (model_path / "pytorch_model.bin").exists():
                checkpoint_path = model_path / "pytorch_model.bin"
            elif (model_path / "model.safetensors").exists():
                checkpoint_path = model_path / "model.safetensors"

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find model checkpoint in {model_path}. "
                "Expected pytorch_model.bin or model.safetensors in the directory or checkpoint subdirectories."
            )

        # Load state_dict based on file type
        if checkpoint_path.suffix == ".safetensors" or str(checkpoint_path).endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(checkpoint_path)
            # Move tensors to device
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle state_dict that might be wrapped in 'model' key
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        nrms_net.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded model from {checkpoint_path}")
    else:
        # Assume it's a direct state_dict file
        # Load state_dict based on file type
        if model_path.suffix == ".safetensors" or str(model_path).endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path)
            # Move tensors to device
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        else:
            state_dict = torch.load(model_path, map_location=device)

        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        nrms_net.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded model from {model_path}")

    return nrms_net


def evaluate_test(
    model_path: str | pathlib.Path,
    pretrained: str,
    history_size: int,
    max_len: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start Evaluation on Test Dataset")

    """
    0. Define Parameters & Functions
    """
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)

    """
    1. Load Pretrained Model
    """
    logging.info("Loading Pretrained Model")
    nrms_net = load_pretrained_model(model_path, pretrained, history_size, device)

    """
    2. Load Test Data & Create Dataset
    """
    logging.info("Loading Test Dataset")
    test_news_df = read_news_df(MIND_SMALL_TEST_DATASET_DIR / "news.tsv")
    test_behavior_df = read_behavior_df(MIND_SMALL_TEST_DATASET_DIR / "behaviors.tsv", is_test=True)
    test_dataset = MINDValDataset(test_behavior_df, test_news_df, transform_fn, history_size)

    """
    3. Evaluate Model on Test Dataset
    """
    logging.info("Evaluating Model on Test Dataset")
    metrics = evaluate(nrms_net, test_dataset, device)
    logging.info("Test Metrics:")
    logging.info(metrics.dict())


@hydra.main(version_base=None, config_name="eval_config")
def main(cfg: EvalConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)

        # Model path should be provided via hydra config or command line
        # Default to MODEL_OUTPUT_DIR if not specified
        from const.path import MODEL_OUTPUT_DIR

        # Check if model_path is provided, otherwise use most recent model
        if not cfg.model_path or cfg.model_path == "":
            # Try to find the most recent model directory
            if MODEL_OUTPUT_DIR.exists():
                model_dirs = [d for d in MODEL_OUTPUT_DIR.iterdir() if d.is_dir()]
                if model_dirs:
                    model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    logging.info(f"Using most recent model: {model_path}")
                else:
                    raise ValueError(
                        "No model checkpoint found. Please specify model_path in config or ensure models exist in MODEL_OUTPUT_DIR"
                    )
            else:
                raise ValueError("No model checkpoint found. Please specify model_path in config")
        else:
            model_path = cfg.model_path
            logging.info(f"Using specified model: {model_path}")

        evaluate_test(
            model_path=model_path,
            pretrained=cfg.pretrained,
            history_size=cfg.history_size,
            max_len=cfg.max_len,
        )
    except Exception as e:
        logging.error(e)
        raise


if __name__ == "__main__":
    main()
