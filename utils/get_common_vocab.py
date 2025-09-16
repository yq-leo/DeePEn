import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(sys.path)
from src.model_load import load_tokenizer, load_model_only
from src.transfer_matrix.common_vocabulary import CommonVocabulary
from src.transfer_matrix.transfer_matrix import ProbabilityTransferMatrix

# model_paths = [
#     "01-ai/Yi-6B",
#     "Skywork/Skywork-13B-base",
#     "mistralai/Mixtral-8x7B-v0.1",
#     "meta-llama/Llama-2-70b-hf",
#     "TigerResearch/tigerbot-13b-base-v2",
#     "mistralai/Mistral-7B-v0.1",
#     "internlm/internlm-20b",
#     "meta-llama/Llama-2-13b-hf"
# ]

probability_transfer_matrix_save_path = sys.argv[1] + "/"
model_paths = sys.argv[2:]
probability_transfer_matrix_name_list = [os.path.basename(model_path) for model_path in model_paths]
probability_transfer_matrix_save_path += "_".join(probability_transfer_matrix_name_list)
print("probability_transfer_matrix_save_path:", probability_transfer_matrix_save_path)
temperature = 100

tokenizers = [load_tokenizer(model_path) for model_path in model_paths]
vocab_lengths = [len(tokenizer.get_vocab()) for tokenizer in tokenizers]
print(vocab_lengths)

common_vocabulary = CommonVocabulary(*tokenizers)
common_vocab_list = common_vocabulary.get_common_vocab_list(*common_vocabulary.vocabs)
print(f"common_vocab_list: {len(common_vocab_list)}")