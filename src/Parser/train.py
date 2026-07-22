from tokenizer import build_vocab, build_entity_vocab, build_reverse_entity_vocab, encode,
import json
import torch
import torch.nn as nn
# Build vocab
with open("training_dataset.json","r") as f:
     Dataset = json.load(f)
vocab = build_vocab(Dataset)
subject_vocab=build_entity_vocab(Dataset, "subject")
relation_vocab=build_entity_vocab(Dataset, "relation")
object_vocab=build_entity_vocab(Dataset, "object")

re_subject_vocab = build_reverse_entity_vocab(Dataset, "subject")
re_relation_vocab = build_reverse_entity_vocab(Dataset, "relation")
re_object_vocab = build_reverse_entity_vocab(Dataset, "object")
