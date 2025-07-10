GANGlowFace is a custom-built Generative Adversarial Network (GAN) for synthesizing high-resolution facial images from latent vectors. It was designed as part of my submission for a generative imaging role, showcasing core skills in GAN architecture, training, and image synthesis.

🔧 Project Features

- Custom Generator and Discriminator written in PyTorch
- Trained on a small real-world dataset of aligned facial images (64×64 resolution)
- Uses Binary Cross-Entropy Loss for adversarial training
- Produces clean sample outputs saved every 10 epochs
- Designed to be easily extended into StyleGAN or latent space editing

Training Details

- Latent vector size: 100
- Optimizer: Adam (LR = 0.0002, betas = 0.5, 0.999)
- Epochs: 100
- Batch size: 4
- Dataset Format: `ImageFolder` with at least one subfolder (e.g. `fake_class`)
- Output folder: `outputs/generated/`
