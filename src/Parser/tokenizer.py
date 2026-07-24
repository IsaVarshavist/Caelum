
def build_vocab(dataset):
    tokendict={"<PAD>":0,"<UNK>":1}
    for item in dataset:
       sentence = item["input"]
       sentence = sentence.lower()
       sentence = sentence.replace(".","")
       sentence = sentence.replace(",","")
       tokens = sentence.split()
       for token in tokens:
          if token not in tokendict:
             tokendict[token]=len(tokendict)
    return tokendict
def build_entity_vocab(dataset, entity_key):
    vocab={}
    for item in dataset:
       entity = item[entity_key]
       if entity not in vocab:
          vocab[entity]=len(vocab)
    return vocab

def build_reverse_entity_vocab(dataset, entity_key):
    vocab={}
    for item in dataset:
        entity = item[entity_key]
        if entity not in vocab:
           vocab[len(vocab)]=entity
    return vocab
def encode(sentence, vocab):
    sentence = sentence.lower()
    sentence = sentence.replace(".","")
    sentence = sentence.replace(",","")
    sentence = sentence.replace("'s","")
    tokens = sentence.split()
    encoding=[]
    for item in tokens:
        encoding.append(vocab.get(item, vocab["<UNK>"]))
    return encoding
def encode_dataset(dataset, vocab, subject_vocab, relation_vocab, object_vocab):
    encoded = []
    for item in dataset:
       encoded.append({
            "input": encode(item["input"], vocab),
            "subject":subject_vocab[item["subject"]],
            "relation":relation_vocab[item["relation"]],
            "object":object_vocab[item["object"]]})
    return encoded
def max_length(dataset, vocab):
    max_length=0
    for item in dataset:
        encoded = encode(item["input"], vocab)
        if len(encoded) > max_length:
           max_length = len(encoded)
    return max_length
def pad(sequence, max_length, pad_id):
    while len(sequence) < max_length:
          sequence.append(pad_id)
    return sequence

def normalize_encoding(dataset, vocab, subject_vocab, relation_vocab, object_vocab):
    max_len= max_length(dataset, vocab)
    encoding = encode_dataset(dataset, vocab, subject_vocab, relation_vocab, object_vocab)
    normalized_encoding=[]
    for item in encoding:
        normalized_encoding.append({"input":pad(item["input"],max_len,vocab["<PAD>"]),"subject":item["subject"],
        "relation":item["relation"],
        "object":item["object"]})
    return normalized_encoding
