import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224
NUM_OBJECTS = 12

OBJECT_DIM = 256
QUESTION_DIM = 768
HIDDEN_DIM = 256
NUM_HEADS = 8

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10

LAMBDA_CF = 1.0
LAMBDA_ACYC = 0.1
LAMBDA_SPARSE = 0.2
LAMBDA_CONS = 0.1

EDGE_THRESHOLD = 0.5

DETR_MODEL = "facebook/detr-resnet-50"
BERT_MODEL = "bert-base-uncased"
# Version of the cached DETR features (train.py FeatureCache).
# 1 = before the query-alignment fix (each detection got the embedding of the wrong
#     DETR query, and the layernorm placement depended on the transformers version)
# 2 = embedding of the detection's own query, taken from DETR's last_hidden_state
FEATURE_CACHE_VERSION = 2
