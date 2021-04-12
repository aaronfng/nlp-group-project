import torch.nn as nn


class BasicAutoencoder(nn.Module):
    """ doesn't work. """
    def __init__(self, vocab_size, hidden_dim=64):
        super(BasicAutoencoder, self).__init__()

        # Very simple encoder/decoder at first
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        # Here text is (batch_size, vocab_size) BoW representation.
        out = self.encoder(text)
        out = self.decoder(out)

        return out