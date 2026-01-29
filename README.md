# Homework 3 - AI Powered Image Generation by GAN

## 🚀 Project Overview
This project implements a Discord bot capable of generating images based on text input. It features a custom-trained **DCGAN** model optimized for horse images using the CIFAR-10 dataset.

### 🏁 Deliverables
- **GitHub Repository**: [https://github.com/wjzdev-dotcom/Homework3-GAN-Bot]
- **Discord Bot**: HelloWorld#7424 (ID: 1461092058621149297)

---

## 🛠️ Implementation Details
- ![GAN Result](./horse.png)
### 1. Model Development (GAN)
- **Architecture**: A Deep Convolutional GAN (DCGAN).
- **Optimization Strategy**: To improve image fidelity within a limited training window, I implemented a **Class-Specific Training** approach. I filtered the CIFAR-10 dataset to include only the **Horse (Class 7)** category.
- **Training**: 50 epochs on an NVIDIA TITAN RTX.

### 2. Bot Integration
- **Framework**: Developed using `discord.py`.
- **Logic**: The bot monitors for the keyword "horse". Upon detection, it triggers the GAN's latent space sampling to generate a 64x64 synthetic image.
- **Security**: Following DevOps best practices, the Discord Token is managed via **OS Environment Variables** to prevent credential leakage.

---

## ⚡ Challenges & Solutions
- **Environment Conflict**: Resolved a critical `torchvision::nms` operator error caused by local package interference in `~/.local` by using `PYTHONNOUSERSITE=1`.
- **Storage Quota**: Overcame a "No space left on device" error during installation by redirecting the pip cache to a temporary data drive partition.

---

## ⚖️ Ethical Considerations
While generating animal imagery is benign, the underlying technology could be exploited for creating **Deepfakes** if trained on human datasets. To mitigate such risks, developers should implement mandatory digital watermarking and clear labeling for AI-generated content to maintain transparency and prevent misinformation.

---

## 🤖 LLM Usage Statement
I utilized **Gemini** as a high-precision engineering collaborator to:
1. Debug complex environment pathing issues in Linux.
2. Draft the initial asynchronous framework for the Discord bot.
3. Optimize the GAN's data loader for sub-class training.
