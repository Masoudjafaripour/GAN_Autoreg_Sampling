import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Settings
z_dim = 10
seq_len = 10  # length of sequence
n_samples = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# GAN Generator
# ----------------------------
class TinyGANGenerator(nn.Module):
    def __init__(self, z_dim, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, seq_len), nn.Sigmoid()  # output sequence between 0 and 1
        )

    def forward(self, z):
        return self.net(z)

# ----------------------------
# Autoregressive Model
# ----------------------------
class TinyAutoRegressive(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.net = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.output = nn.Linear(32, 1)

    def forward(self, x, h=None):
        out, h = self.net(x, h)
        out = self.output(out)
        return out, h

# ----------------------------
# Instantiate models
# ----------------------------
gan_gen = TinyGANGenerator(z_dim, seq_len).to(device)
auto_reg = TinyAutoRegressive(seq_len).to(device)

# ----------------------------
# Sampling functions
# ----------------------------
def sample_from_gan(model, n_samples):
    z = torch.randn(n_samples, z_dim).to(device)
    with torch.no_grad():
        samples = model(z)
    return samples.cpu().numpy()

def sample_from_autoregressive(model, n_samples, seq_len):
    samples = []
    model.eval()
    for _ in range(n_samples):
        seq = []
        input_token = torch.zeros(1, 1, 1).to(device)  # Start with zero
        h = None
        for _ in range(seq_len):
            out, h = model(input_token, h)
            prob = torch.sigmoid(out[:, -1, :])
            token = torch.bernoulli(prob)  # Sample 0 or 1
            seq.append(token.item())
            input_token = token.view(1, 1, 1)
        samples.append(seq)
    return np.array(samples)

# ----------------------------
# Sampling
# ----------------------------
gan_samples = sample_from_gan(gan_gen, n_samples)
auto_samples = sample_from_autoregressive(auto_reg, n_samples, seq_len)

# ----------------------------
# Visualization
# ----------------------------
fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
for i in range(n_samples):
    axes[0, i].imshow(gan_samples[i].reshape(1, -1), cmap='gray', aspect='auto')
    axes[0, i].set_title(f'GAN Sample {i+1}')
    axes[0, i].axis('off')

    axes[1, i].imshow(auto_samples[i].reshape(1, -1), cmap='gray', aspect='auto')
    axes[1, i].set_title(f'AR Sample {i+1}')
    axes[1, i].axis('off')

plt.suptitle('GAN vs Autoregressive Sampling')
plt.tight_layout()
plt.show()

# ----------------------------
# Short Summary
# ----------------------------
print("\nSummary:")
print("GAN samples diversity comes from different random latent vectors z.")
print("Autoregressive samples diversity comes from random choices at each step.")
