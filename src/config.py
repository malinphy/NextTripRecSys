
import os
import torch

class Config:

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

    ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'original_data')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    

    os.makedirs(ORIGINAL_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    

    RAW_TRAIN = os.path.join(ORIGINAL_DATA_DIR, 'train_set.csv')
    RAW_TEST = os.path.join(ORIGINAL_DATA_DIR, 'test_set.csv')
    RAW_GT = os.path.join(ORIGINAL_DATA_DIR, 'ground_truth.csv')
    

    TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_fe_eng.csv')
    TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'final_test.csv')
    

    PATH_PREFIX = os.path.join(CHECKPOINT_DIR, "checkpoint")


    WINDOW_SIZE = 5


    CITY_EMBEDDING_DIM = 128
    COUNTRY_EMBEDDING_DIM = 64
    HIDDEN_DIM = 512
    NUM_RESIDUAL_BLOCKS = 3
    NUM_NUMERIC_FEATURES = 4
    DROPOUT = 0.3


    LEARNING_RATE = 0.001
    NUM_EPOCHS = 25
    BATCH_SIZE = 1024
    WEIGHT_DECAY = 1e-5

    NUM_CITY = 39901   
    NUM_COUNTRY = 195   
    IGNORE_INDEX = 0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
