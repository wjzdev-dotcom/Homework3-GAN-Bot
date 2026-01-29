import sys
import os
# 屏蔽本地包干扰
bad_path = "/home/jingzhi/.local/lib/python3.10/site-packages"
if bad_path in sys.path:
    sys.path.remove(bad_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset 
import numpy as np

# 参数
device = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
Z_DIM = 100
EPOCHS = 50 
FEATURES = 64

# 数据准备
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

full_dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transform, download=True)

# === 关键修改：只保留“马” (Class Index 7) ===
horse_indices = [i for i, label in enumerate(full_dataset.targets) if label == 7]
horse_dataset = Subset(full_dataset, horse_indices)
loader = DataLoader(horse_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 网络结构
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, FEATURES, 4, 2, 1), nn.LeakyReLU(0.2),
            self._block(FEATURES, FEATURES*2, 4, 2, 1),
            self._block(FEATURES*2, FEATURES*4, 4, 2, 1),
            self._block(FEATURES*4, FEATURES*8, 4, 2, 1),
            nn.Conv2d(FEATURES*8, 1, 4, 2, 0), nn.Sigmoid(),
        )
    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2),
        )
    def forward(self, x): return self.disc(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(Z_DIM, FEATURES*16, 4, 1, 0),
            self._block(FEATURES*16, FEATURES*8, 4, 2, 1),
            self._block(FEATURES*8, FEATURES*4, 4, 2, 1),
            self._block(FEATURES*4, FEATURES*2, 4, 2, 1),
            nn.ConvTranspose2d(FEATURES*2, 3, 4, 2, 1), nn.Tanh(),
        )
    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(),
        )
    def forward(self, x): return self.gen(x)

# 训练流程
gen = Generator().to(device)
disc = Discriminator().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
criterion = nn.BCELoss()

print("🚀 开始训练“马匹生成器” (Target: Horse Only)...")

for epoch in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(len(real), Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Train Disc
        disc_real = disc(real).reshape(-1)
        loss_d_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2
        disc.zero_grad(); loss_d.backward(); opt_disc.step()

        # Train Gen
        output = disc(fake).reshape(-1)
        loss_g = criterion(output, torch.ones_like(output))
        gen.zero_grad(); loss_g.backward(); opt_gen.step()

    print(f"Epoch [{epoch}/{EPOCHS}] Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

torch.save(gen.state_dict(), "horse_generator.pth")
print("✅ 专精马匹模型已保存: horse_generator.pth")