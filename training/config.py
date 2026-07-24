from pathlib import Path

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "unified_dataset.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen_multitask_lora"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Model
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# =====================================================
# Training
# =====================================================

MAX_LENGTH = 1024

NUM_EPOCHS = 2

LEARNING_RATE = 2e-4

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 8

WARMUP_RATIO = 0.03

WEIGHT_DECAY = 0.01

SEED = 42

LOG_LEVEL = "info"

# =====================================================
# LoRA
# =====================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# =====================================================
# Saving
# =====================================================

SAVE_STEPS = 500

LOGGING_STEPS = 50

# =====================================================
# Mixed Precision
# =====================================================

FP16 = True
BF16 = False

# =====================================================
# Logging
# =====================================================

REPORT_TO = "none"

# =====================================================
# Checkpoints
# =====================================================

SAVE_TOTAL_LIMIT = 2

# =====================================================
# Data Loader
# =====================================================

NUM_WORKERS = 2