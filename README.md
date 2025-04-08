<p align="center">
  <picture>
    <source srcset="assets/diffformer-dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="assets/diffformer-light.png" media="(prefers-color-scheme: light)">
    <img src="assets/diffformer_dark.png" alt="Diffformer Logo" width="300">
  </picture>
</p>

Diffformer is a diffusion-based language model that predicts tokens by denoising in embedding space. Instead of directly predicting and autoregressing logits, Diffformer gradually refines the pre-logit embeddings before projecting to logits and sampling.



---

## 🔧 Setup (via [PDM](https://pdm-project.org/en/latest/))


```bash
pdm install
```

To run the training script:

```bash
pdm run python train.py
```

To recreate graphs and charts

```bash
pdm run python test.py
```

