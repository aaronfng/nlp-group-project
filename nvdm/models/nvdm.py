"""
PyTorch implementation of NVDM paper:
Neural Variational Inference for Text Processing. Miao et al. ICML 2016.

https://arxiv.org/pdf/1511.06038.pdf
"""
import torch
import torch.nn as nn


def kl_div(mu, log_sigma):
    """ KL divergence as defined in the paper, given the mu and log(sigma)
    parameters of the Gaussian variational distribution. """
    # In the paper they have an undefined "K" variable, but is set to one
    # in the author's code.
    K = 1
    # logdet(diag(s^2)) = log(prod_i s_i^2) = sum_i log s_i^2 = 2 sum_i log s_i
    return -0.5 * torch.sum(K - mu.pow(2) + 2 * log_sigma - torch.exp(2 * log_sigma))


class NVDM(nn.Module):
    """ Neural Variational Document Model -- BOW VAE. """
    def __init__(self, vocab_size, n_hidden, n_topic, n_sample, device, learn_embeddings=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic

        # Not used for now, just 1 sample on each call to self.decode()
        self.n_sample = n_sample

        # Inference network
        # See paper appendix B for details

        # Embedded Bag-of-Words
        # This converts variable-length sentences into BOW representations
        # Disable learning word embeddings for now.
        self.embed_bow = nn.EmbeddingBag(vocab_size, vocab_size, mode="sum")

        if not learn_embeddings:
            # Convert "embeddings" into non-trainable identity matrix
            # (i.e. word embeddings are always one-hot)
            # So this effectively becomes a CountVectorizer.
            self.embed_bow.requires_grad_(False)
            assert not self.embed_bow.weight.requires_grad
            assert self.embed_bow.weight.size() == (vocab_size, vocab_size)

            self.embed_bow.weight.zero_()
            self.embed_bow.weight.fill_diagonal_(1.0)

        # Encoder: takes the bag-of-words representation of a document
        # and generates an intermediate/hidden representation.
        # Original paper uses ReLU, but that results in values being too large
        # (many NaNs)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
        )

        # Transform the hidden representation to get the
        # Gaussian mu and log-standard deviation variational parameters.
        self.mu = nn.Linear(n_hidden, n_topic)
        self.log_sigma = nn.Linear(n_hidden, n_topic)
        # Author's code initializes weights and biases of log sigma to 0
        # torch.nn.init.zeros_(self.log_sigma.bias)
        # torch.nn.init.zeros_(self.log_sigma.weight)
        # h ~ N(mu, sigma^2)

        # Generative model
        # This linearly transforms the hidden representation
        # before computing softmax probabilities over vocabulary
        # This becomes the "learned embedding"
        self.decoder = nn.Sequential(
            nn.Linear(n_topic, vocab_size)
        )

        # Torch device (GPU/CPU)
        self.device = device

    def encode(self, text, offsets):
        # Convert text into BoW representations.
        X_bow = self.embed_bow(text, offsets)

        # Compute Gaussian parameters
        pi = self.encoder(X_bow)
        mu = self.mu(pi)
        log_sigma = self.log_sigma(pi)
        return X_bow, mu, log_sigma

    def decode(self, X_bow, mu, log_sigma):
        # Take samples from transformed standard Normal
        batch_size = X_bow.size(0)

        # Only 1 sample
        eps = torch.randn((batch_size, self.n_topic), device=self.device)
        doc_vec = torch.exp(log_sigma) * eps + mu

        # Logit probabilities over vocabulary
        logits = torch.log_softmax(self.decoder(doc_vec), dim=1)
        return logits

    def forward(self, text, offsets, kl_weight=1.0):
        """ Here we compute both the logits and total loss. """
        X_bow, mu, log_sigma = self.encode(text, offsets)
        logits = self.decode(X_bow, mu, log_sigma)

        # Reconstruction loss
        # Negative log likelihood,
        # but each logit p(x_i) is weighted by the number
        # of occurrence of x_i in the text, (BoW frequencies).
        loss_rec = -(X_bow * logits).sum(dim=1).mean()

        # KL divergence loss
        loss_kl = kl_div(mu, log_sigma)

        # Possible improvement: can add a hyperparameter (beta)
        # control the weighting of KL.
        # e.g. loss_total = loss_rec + b * loss_kl
        loss_total = loss_rec + kl_weight * loss_kl
        return logits, {"rec": loss_rec, "kl": loss_kl, "total": loss_total}
