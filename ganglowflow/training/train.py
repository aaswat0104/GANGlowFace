import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.generator import Generator
from model.discriminator import Discriminator


# ------------------------------
# 🧠 Configuration
latent_dim = 100
img_size = 64
batch_size = 4
epochs = 100
lr = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 🖼️ Dataset Setup (ImageFolder expects a subfolder)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Scale to [-1, 1] for Tanh
])

dataset = datasets.ImageFolder(root='data/ffhq_sample/', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# 🧠 Model Initialization
G = Generator(latent_dim=latent_dim, img_size=img_size).to(device)
D = Discriminator(img_size=img_size).to(device)

# ------------------------------
# 🎯 Loss & Optimizers
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ------------------------------
# 📂 Output Directories
os.makedirs("outputs/generated", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ------------------------------
# 🔁 Training Loop
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(loader):
        real_imgs = real_imgs.to(device)
        valid = torch.ones((real_imgs.size(0), 1), device=device)
        fake = torch.zeros((real_imgs.size(0), 1), device=device)

        # ---- Train Generator ----
        opt_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        gen_imgs = G(z)
        g_loss = criterion(D(gen_imgs), valid)
        g_loss.backward()
        opt_G.step()

        # ---- Train Discriminator ----
        opt_D.zero_grad()
        real_loss = criterion(D(real_imgs), valid)
        fake_loss = criterion(D(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

    # Save generated samples every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        sample_path = f"outputs/generated/epoch_{epoch+1}.png"
        utils.save_image(gen_imgs[:8], sample_path, normalize=True)
        print(f"Saved sample: {sample_path}")

    print(f"[Epoch {epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

# ------------------------------
# 💾 Save Final Weights
torch.save(G.state_dict(), "checkpoints/generator.pth")
torch.save(D.state_dict(), "checkpoints/discriminator.pth")
