<div align="center">
    <img height=200 src="./.github/images/news-logo.png" alt="News Contents on Smartphone">
</div>

<h1 align="center">📰 Hệ thống Gợi ý Tin tức sử dụng LLM</h1>
<p align="center"><strong>News Recommendation System sử dụng Pre-trained Large Language Model (BERT/DistilBERT) với PyTorch 🚀</strong></p>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt nhanh](#-cài-đặt-nhanh)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
  - [1. Tải dataset](#1-tải-dataset)
  - [2. Train model](#2-train-model)
  - [3. Test/Evaluate model](#3-testevaluate-model)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cấu hình](#-cấu-hình)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Giới thiệu

Dự án này triển khai hệ thống gợi ý tin tức sử dụng **Neural News Recommendation with Multi-Head Self-Attention (NRMS)** kết hợp với các mô hình ngôn ngữ lớn như **BERT** và **DistilBERT**.

### ✨ Tính năng chính

- ✅ Sử dụng Pre-trained Language Models (BERT/DistilBERT) để mã hóa nội dung tin tức
- ✅ Mô hình NRMS với Multi-Head Self-Attention
- ✅ Hỗ trợ dataset MIND (Microsoft News Dataset)
- ✅ Đánh giá với các metrics: AUC, MRR, nDCG@5, nDCG@10

---

## 💻 Yêu cầu hệ thống

### Phần mềm cần thiết

- **Python**: 3.11.3
- **PyTorch**: 2.0.1+
- **CUDA**: Khuyến nghị (nếu có GPU)
- **Rye**: Package manager (hoặc pip)

### Phần cứng khuyến nghị

- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị cho training)
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **Disk**: Tối thiểu 10GB cho dataset và model

---

## 🚀 Cài đặt nhanh

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd kaggle-news-recommendation
```

### Bước 2: Cài đặt dependencies

**Cách 1: Sử dụng Rye (khuyến nghị)**

```bash
# Cài đặt Rye (nếu chưa có)
curl -sSf https://rye-up.com/get | bash

# Đồng bộ dependencies
rye sync
```

**Cách 2: Sử dụng pip**

```bash
pip install -r requirements.txt
```

### Bước 3: Thiết lập môi trường

```bash
# Thiết lập PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Hoặc thêm vào ~/.bashrc hoặc ~/.zshrc để tự động load
echo 'export PYTHONPATH=$(pwd)/src:$PYTHONPATH' >> ~/.bashrc
```

### Bước 4: Kiểm tra cài đặt

```bash
# Kiểm tra Python version
python --version  # Nên là 3.11.3

# Kiểm tra PyTorch và CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📖 Hướng dẫn sử dụng

### 1. Tải Dataset

Trước khi train, bạn cần tải dataset MIND:

```bash
# Sử dụng Rye
rye run python ./dataset/download_mind.py

# Hoặc sử dụng Python trực tiếp (sau khi đã set PYTHONPATH)
python ./dataset/download_mind.py
```

**Lưu ý**: Quá trình download có thể mất vài phút tùy vào tốc độ mạng.

Sau khi tải xong, cấu trúc thư mục sẽ như sau:

```
dataset/
└── mind/
    ├── small/
    │   ├── train/
    │   │   ├── behaviors.tsv
    │   │   └── news.tsv
    │   └── val/
    │       ├── behaviors.tsv
    │       └── news.tsv
    └── large/
        ├── train/
        ├── val/
        └── test/
```

---

### 2. Train Model

#### 🎯 Train với cấu hình mặc định

Cách đơn giản nhất để bắt đầu train:

```bash
# Sử dụng Rye
rye run python src/experiment/train.py

# Hoặc sử dụng Python trực tiếp
python src/experiment/train.py
```

#### ⚙️ Train với cấu hình tùy chỉnh

Bạn có thể tùy chỉnh các hyperparameters qua command line:

```bash
rye run python src/experiment/train.py \
    random_seed=42 \
    pretrained="distilbert-base-uncased" \
    npratio=4 \
    history_size=50 \
    batch_size=16 \
    gradient_accumulation_steps=8 \
    epochs=3 \
    learning_rate=1e-4 \
    weight_decay=0.0 \
    max_len=30
```

#### 📝 Giải thích các tham số

| Tham số                       | Mô tả                                     | Giá trị mặc định            |
| ----------------------------- | ----------------------------------------- | --------------------------- |
| `random_seed`                 | Seed cho reproducibility                  | `42`                        |
| `pretrained`                  | Tên mô hình pre-trained                   | `"distilbert-base-uncased"` |
| `npratio`                     | Tỷ lệ negative sampling                   | `4`                         |
| `history_size`                | Số lượng tin tức trong lịch sử người dùng | `50`                        |
| `batch_size`                  | Batch size cho training                   | `16`                        |
| `gradient_accumulation_steps` | Số bước tích lũy gradient                 | `8`                         |
| `epochs`                      | Số epochs để train                        | `1`                         |
| `learning_rate`               | Learning rate                             | `1e-4`                      |
| `weight_decay`                | Weight decay cho regularization           | `0.0`                       |
| `max_len`                     | Độ dài tối đa của sequence                | `30`                        |

**Lưu ý**:

- Batch size thực tế = `batch_size × gradient_accumulation_steps` = 16 × 8 = 128
- Model sẽ được lưu tự động trong `output/model/` với timestamp

#### 📊 Theo dõi quá trình training

- Logs được lưu trong `output/log/`
- Model checkpoints được lưu trong `output/model/YYYY-MM-DD/HH-MM-SS/checkpoint-{step}/`
- Mỗi checkpoint chứa:
  - `model.safetensors`: Trọng số của model
  - `optimizer.pt`: Trạng thái optimizer
  - `scheduler.pt`: Trạng thái learning rate scheduler
  - `trainer_state.json`: Trạng thái trainer

---

### 3. Test/Evaluate Model

#### 🧪 Đánh giá trên Test Dataset

Sau khi train xong, bạn có thể đánh giá model trên test dataset:

```bash
# Sử dụng model mới nhất (tự động tìm)
rye run python src/experiment/evaluate.py

# Hoặc chỉ định đường dẫn model cụ thể
rye run python src/experiment/evaluate.py \
    model_path="output/model/2025-11-28/09-06-09/checkpoint-614" \
    pretrained="distilbert-base-uncased" \
    history_size=50 \
    max_len=30
```

#### 📈 Các metrics được đánh giá

- **AUC**: Area Under the ROC Curve
- **MRR**: Mean Reciprocal Rank
- **nDCG@5**: Normalized Discounted Cumulative Gain tại top 5
- **nDCG@10**: Normalized Discounted Cumulative Gain tại top 10

#### 🎲 So sánh với Random Baseline

Bạn cũng có thể chạy random baseline để so sánh:

```bash
rye run python src/experiment/evaluate_random.py
```

---

## 📁 Cấu trúc dự án

```
kaggle-news-recommendation/
├── 📂 dataset/                    # Thư mục chứa dataset
│   ├── download_mind.py          # Script tải MIND dataset
│   └── mind/                      # Dataset MIND sau khi tải
│       ├── small/                 # MIND Small dataset
│       └── large/                 # MIND Large dataset
│
├── 📂 src/                        # Source code chính
│   ├── 📂 config/                 # Cấu hình
│   │   └── config.py             # TrainConfig và EvalConfig
│   │
│   ├── 📂 const/                  # Constants
│   │   └── path.py               # Đường dẫn các thư mục
│   │
│   ├── 📂 evaluation/            # Đánh giá model
│   │   └── RecEvaluator.py       # Metrics: AUC, MRR, nDCG
│   │
│   ├── 📂 experiment/            # Scripts train và evaluate
│   │   ├── train.py              # 🚂 Script training
│   │   ├── evaluate.py           # 🧪 Script evaluation
│   │   └── evaluate_random.py    # Random baseline
│   │
│   ├── 📂 mind/                  # Xử lý MIND dataset
│   │   ├── dataframe.py          # Đọc dữ liệu từ TSV
│   │   └── MINDDataset.py        # Dataset class cho PyTorch
│   │
│   ├── 📂 recommendation/        # Mô hình recommendation
│   │   └── 📂 nrms/              # NRMS model
│   │       ├── NRMS.py           # Model chính
│   │       ├── PLMBasedNewsEncoder.py  # Encoder cho tin tức
│   │       ├── UserEncoder.py    # Encoder cho người dùng
│   │       └── AdditiveAttention.py   # Attention mechanism
│   │
│   └── 📂 utils/                 # Utilities
│       ├── logger.py             # Logging
│       ├── path.py               # Path utilities
│       ├── random_seed.py        # Set random seed
│       └── text.py               # Text processing
│
├── 📂 output/                     # Output files
│   ├── 📂 model/                 # Saved models
│   └── 📂 log/                   # Training logs
│
├── 📂 test/                       # Unit tests
│
├── 📄 requirements.txt           # Python dependencies
├── 📄 pyproject.toml             # Rye configuration
└── 📄 README.md                  # File này
```

---

## ⚙️ Cấu hình

### Cấu hình Training

File cấu hình: `src/config/config.py`

```python
@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "distilbert-base-uncased"  # hoặc "bert-base-uncased"
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_len: int = 30
```

### Cấu hình Evaluation

```python
@dataclass
class EvalConfig:
    random_seed: int = 42
    pretrained: str = "distilbert-base-uncased"
    history_size: int = 50
    max_len: int = 30
    model_path: str = ""  # Để trống sẽ dùng model mới nhất
```

---

## 📊 Kết quả thực nghiệm

### Kết quả trên MIND Small Dataset

| Model                      | AUC       | MRR       | nDCG@5    | nDCG@10   | Thời gian train |
| -------------------------- | --------- | --------- | --------- | --------- | --------------- |
| Random Recommendation      | 0.500     | 0.201     | 0.203     | 0.267     | -               |
| **NRMS + DistilBERT-base** | **0.674** | **0.297** | **0.322** | **0.387** | **15.0h**       |
| **NRMS + BERT-base**       | **0.689** | **0.306** | **0.336** | **0.400** | **28.5h**       |

_Kết quả được đo trên Single GPU (V100 x 1)_

### Model đã được train sẵn

Nếu bạn muốn sử dụng model đã được train sẵn:

| Model                  | Link                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| NRMS + DistilBERT-base | [Google Drive](https://drive.google.com/file/d/1cw9WQSOVYJdYJCuIrSmU8odV2nsmith5/view?usp=sharing) |
| NRMS + BERT-base       | [Google Drive](https://drive.google.com/file/d/1ARiUgSVwcDFopFoIusp2MGQzwTMncOFf/view?usp=sharing) |

---

## 🔧 Troubleshooting

### ❌ Lỗi: `ModuleNotFoundError`

**Nguyên nhân**: Chưa set PYTHONPATH

**Giải pháp**:

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### ❌ Lỗi: `CUDA out of memory`

**Nguyên nhân**: GPU không đủ bộ nhớ

**Giải pháp**:

- Giảm `batch_size` (ví dụ: từ 16 xuống 8)
- Tăng `gradient_accumulation_steps` để giữ nguyên effective batch size
- Sử dụng `torch.bfloat16` (đã được enable mặc định)

### ❌ Lỗi: Dataset không tìm thấy

**Nguyên nhân**: Chưa tải dataset hoặc đường dẫn sai

**Giải pháp**:

```bash
# Kiểm tra dataset đã tải chưa
ls dataset/mind/small/train/

# Nếu chưa có, chạy lại script download
python ./dataset/download_mind.py
```

### ❌ Lỗi: Model checkpoint không tìm thấy khi evaluate

**Nguyên nhân**: Đường dẫn model sai hoặc chưa train

**Giải pháp**:

- Kiểm tra model đã được lưu trong `output/model/`
- Chỉ định đường dẫn đầy đủ trong config hoặc command line
- Đảm bảo checkpoint có file `model.safetensors` hoặc `pytorch_model.bin`

### ⚠️ Training chậm

**Các cách tối ưu**:

- Sử dụng GPU thay vì CPU
- Sử dụng `distilbert-base-uncased` thay vì `bert-base-uncased` (nhanh hơn ~2x)
- Tăng `batch_size` và `gradient_accumulation_steps` nếu GPU đủ mạnh

---

## 📚 Tài liệu tham khảo

### Papers

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**

   - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.
   - https://aclanthology.org/N19-1423

2. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**

   - Sanh, V., Debut, L., Chaumond, J., & Wolf, T.
   - https://arxiv.org/abs/1910.01108

3. **Neural News Recommendation with Multi-Head Self-Attention**

   - Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., & Xie, X.
   - https://aclanthology.org/D19-1671

4. **Empowering News Recommendation with Pre-Trained Language Models**

   - Wu, C., Wu, F., Qi, T., & Huang, Y.
   - https://doi.org/10.1145/3404835.3463069

5. **MIND: A Large-scale Dataset for News Recommendation**
   - Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J., Wu, W., & Zhou, M.
   - https://aclanthology.org/2020.acl-main.331

---

## 📝 Citation

Nếu bạn sử dụng code này trong nghiên cứu, vui lòng cite:

```bibtex
@misc{yuki-yada-news-rec,
  author = {Yuki Yada},
  title = {News Recommendation using PLMs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YadaYuki/news-recommendation-llm}}
}
```

---

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Nếu bạn muốn sử dụng phần mềm này trong nghiên cứu hoặc dự án, vui lòng liên hệ: yada.yuki@fuji.waseda.jp

---

## 📄 License

Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

<div align="center">
    <p>⭐ Nếu project này hữu ích, hãy star để ủng hộ! ⭐</p>
</div>
