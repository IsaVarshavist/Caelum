from tokenizer import build_vocab, build_entity_vocab, build_reverse_entity_vocab, encode, encode_dataset, normalize_encoding
from model import TripleExtractor
import json
import torch
import torch.nn as nn
# Build vocab
with open("training_dataset.json","r") as f:
     Dataset = json.load(f)
vocab = build_vocab(Dataset)
subject_vocab=build_entity_vocab(Dataset, "subject")
print("Subject vocab done!")
relation_vocab=build_entity_vocab(Dataset, "relation")
print("relation vocab done!")
object_vocab=build_entity_vocab(Dataset, "object")
print("object vocab done!")
re_subject_vocab = build_reverse_entity_vocab(Dataset, "subject")
print("Reverse subject vocab done")
re_relation_vocab = build_reverse_entity_vocab(Dataset, "relation")
print("Reverse Relation vocab done")
re_object_vocab = build_reverse_entity_vocab(Dataset, "object")
print("Reverse object vocab done")
model=TripleExtractor(vocab_size=len(vocab),num_subjects=len(subject_vocab),num_relations=len(relation_vocab),num_objects=len(object_vocab))
encoding=normalize_encoding(Dataset, vocab, subject_vocab, relation_vocab, object_vocab)
print("Encoding Layer 1 done")
inputs=[]
for item in encoding:
   inputs.append(item["input"])
x = torch.tensor(inputs)
print("Encoding done, beginning prediction")
subject, relation, object = model(x)
print(subject.shape)
print(relation.shape)
print(object.shape)
