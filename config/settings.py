MODEL_NAME = "google/pegasus-cnn_dailymail"
FINE_TUNED_MODEL_PATH = "./models/pegasus-samsum-model"
TOKENIZER_PATH = "./models/tokenizer"

 
DATASET_PATH = "./data/samsum_dataset"
DATASET_URL = "https://github.com/entbappy/Branching-tutorial/raw/master/summarizer-data.zip"

BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
NUM_EPOCHS = 1   
WARMUP_STEPS = 500   
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 8   
LEARNING_RATE = 5e-5   
MAX_INPUT_LENGTH = 1024   
MAX_TARGET_LENGTH = 128   
SAVE_STEPS = 500   
EVAL_STEPS = 500   
LOGGING_STEPS = 100   

LENGTH_PENALTY = 0.8
NUM_BEAMS = 8
MAX_GENERATION_LENGTH = 128
MIN_GENERATION_LENGTH = 10

SUMMARY_LENGTHS = {
    "short": {"max_tokens": 25},
    "medium": {"max_tokens": 50}, 
    "long": {"max_tokens": 80}
}

LOG_LEVEL = "INFO"
LOG_FILE = "./logs/training.log"
METRICS_FILE = "./logs/metrics.txt"

DEVICE = "cuda"
