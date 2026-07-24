import torch.nn as nn
#Defining model

class TripleExtractor(nn.Module):
      def __init__(self, vocab_size, embedding_dim=64,hidden_dim=128, num_subjects=None,
                   num_relations=None, num_objects=None):
          super().__init__()

          self.embedding = nn.Embedding(vocab_size, embedding_dim)
          self.hidden = nn.Linear(embedding_dim, hidden_dim)
          self.relu = nn.ReLU()
          self.subject_head = nn.Linear(hidden_dim, num_subjects)
          self.relation_head = nn.Linear(hidden_dim, num_relations)
          self.object_head = nn.Linear(hidden_dim, num_objects)

      def forward(self, x):

          x = self.embedding(x)

          x = x.mean(dim=1)      # one 64-d vector per sentence

          x = self.hidden(x)

          x = self.relu(x)

          subject = self.subject_head(x)

          relation = self.relation_head(x)

          object = self.object_head(x)

          return subject, relation, object
