
#Defining model

class TripleExtractor(nn.Module):
      def __init__(self, vocab_size, embedding_dim):
          super().__init__()

          self.embedding = nn.Embedding(vocab_size, embedding_dim)

