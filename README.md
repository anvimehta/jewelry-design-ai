# The Necklace Design Generator

A Variational Autoencoder (VAE) that learns a latent space over necklace images and generates new designs by sampling from it.

## Approach

- **Image sources.** DeepFashion is clothing-focused and doesn't have clean necklace labels, so we use the Kaggle *Fashion Product Images (Small)* dataset and filter `articleType == 'Necklace'` via `styles.csv`. Any folder of necklace images also works — drop them into `data/necklaces/`.
- **Model.** A compact convolutional VAE: 4-stage strided-conv encoder → `mu` / `log_var` heads → reparameterized sample → transposed-conv decoder → sigmoid. Trained on 64×64 RGB images with an MSE reconstruction + KL divergence loss.
- **Exploration.** After training we visualize reconstructions, sample from the prior, interpolate between real designs in latent space, run per-dimension traversals, and sweep `latent_dim` × `learning_rate` to compare variety/quality.

## Quickstart

```bash
# 1. Install deps (Python 3.10+ recommended).
pip install torch torchvision pandas pillow matplotlib numpy tqdm kaggle

# 2. Get data (either Kaggle CLI or drop your own images in data/necklaces/).
mkdir -p data
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data --unzip

# 3. Open the notebook and run top to bottom.
jupyter lab base.ipynb
```

The notebook auto-detects Apple Silicon (`mps`), CUDA, or CPU.

## Layout

- [base.ipynb](base.ipynb) — everything: data, model, training, latent-space exploration, hyperparameter sweep.
- `data/necklaces/` — input images (gitignored).
- `checkpoints/` — saved model weights (gitignored).
