<p align="center">
  <picture>
    <source srcset="assets/flowformer-dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="assets/flowformer-light.png" media="(prefers-color-scheme: light)">
    <img src="assets/flowformer_dark.png" alt="Flowformer Logo" width="300">
  </picture>
</p>

Flowformer is a flow matching-based language model that predicts tokens by integrating in embedding space. Instead of directly predicting and autoregressing logits, Diffformer gradually refines the pre-logit embeddings before projecting to logits and sampling.

## Architecture

![alt text](assets/architecture.svg)


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

