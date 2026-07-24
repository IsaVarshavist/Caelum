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
subject_targets = torch.tensor([item["subject"] for item in encoding])
relation_targets = torch.tensor([item["relation"] for item in encoding])
object_targets = torch.tensor([item["object"] for item in encoding])
print("Encoding done, beginning prediction")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    subject, relation, obj = model(x)
    subject_loss = criterion(subject, subject_targets)
    relation_loss = criterion(relation, relation_targets)
    object_loss = criterion(obj, object_targets)
    loss = subject_loss + relation_loss + object_loss
    loss.backward()
    optimizer.step()
    print(loss.item())

subject, relation, obj = model(x)

subject_pred = subject.argmax(dim=1)

for i in range(5):
    print(
        "Pred:",
        re_subject_vocab[subject_pred[i].item()],
        "| Actual:",
        re_subject_vocab[subject_targets[i].item()]
    )
relation_pred = relation.argmax(dim=1)

for i in range(5):
    print(
        "Pred:",
        re_relation_vocab[relation_pred[i].item()],
        "| Actual:",
        re_relation_vocab[relation_targets[i].item()]
    )
object_pred = obj.argmax(dim=1)

for i in range(5):
    print(
        "Pred:",
        re_object_vocab[object_pred[i].item()],
        "| Actual:",
        re_object_vocab[object_targets[i].item()]
    )
