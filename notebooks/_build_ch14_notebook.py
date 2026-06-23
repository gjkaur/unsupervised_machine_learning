"""
Generate 14_chapter14_conclusion.ipynb — Hands-On Unsupervised Learning Using Python, Chapter 14.
Run: python _build_ch14_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "14_chapter14_conclusion.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}


def code(*lines: str) -> dict:
    source = [line + "\n" for line in lines]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(md(
        "# Chapter 14: Conclusion",
        "",
        "**Book:** *Hands-On Unsupervised Learning Using Python* — Ankur A. Patel (O'Reilly, 2019)  ",
        "**Audience:** Beginners  ",
        "**Goal:** Recap the full course — **Parts I–IV**, key tools, and how unsupervised learning fits into applied machine learning workflows.",
        "",
        "---",
        "",
        "## What you will learn",
        "",
        "1. How the four parts of the book connect into one workflow",
        "2. When to use **scikit-learn (Scientific Kit for Learning)** vs **TensorFlow / Keras**",
        "3. A course-wide glossary of essential terms",
        "4. Reflection questions for your next unsupervised learning project",
        "",
        "> This chapter is **mostly markdown** — one short code cell verifies your environment. No heavy model training.",
    ))

    cells.append(md(
        "## Setup — run this cell first",
        "",
        "Quick **sanity check** — import representative libraries from each part of the course.",
    ))

    cells.append(code(
        "# Import warnings module so we can hide non-critical messages",
        "import warnings",
        "",
        "warnings.filterwarnings(\"ignore\")",
        "",
        "# --- Part I–II: scikit-learn (Scientific Kit for Learning) ecosystem ---",
        "import numpy as np",
        "",
        "import pandas as pd",
        "",
        "from sklearn.decomposition import PCA",
        "",
        "from sklearn.cluster import KMeans, DBSCAN",
        "",
        "from sklearn.ensemble import RandomForestClassifier",
        "",
        "# --- Part III–IV: TensorFlow (deep learning framework) / Keras ---",
        "import tensorflow as tf",
        "",
        "from tensorflow.keras import layers",
        "",
        "# --- Optional libraries used in later chapters ---",
        "try:",
        "",
        "    import lightgbm as lgb",
        "",
        "    _lgb_ok = True",
        "",
        "except ImportError:",
        "",
        "    _lgb_ok = False",
        "",
        "try:",
        "",
        "    import tslearn",
        "",
        "    _tslearn_ok = True",
        "",
        "except ImportError:",
        "",
        "    _tslearn_ok = False",
        "",
        "try:",
        "",
        "    import hdbscan",
        "",
        "    _hdbscan_ok = True",
        "",
        "except ImportError:",
        "",
        "    _hdbscan_ok = False",
        "",
        "# Report environment status",
        "print(\"Course environment sanity check\")",
        "print(f\"  NumPy {np.__version__} | Pandas {pd.__version__} | scikit-learn OK\")",
        "print(f\"  TensorFlow {tf.__version__}\")",
        "print(f\"  LightGBM: {'OK' if _lgb_ok else 'not installed'}\")",
        "print(f\"  tslearn: {'OK' if _tslearn_ok else 'not installed'}\")",
        "print(f\"  hdbscan: {'OK' if _hdbscan_ok else 'not installed'}\")",
        "print(\"\\nAll core imports succeeded — you are ready to apply unsupervised learning!\")",
    ))

    cells.append(md(
        "---",
        "",
        "## Part I — Fundamentals of Unsupervised Learning (Chapters 1–2)",
        "",
        "| Chapter | Topic | Key takeaway |",
        "|---------|-------|--------------|",
        "| **1** | ML (Machine Learning) ecosystem | Unsupervised learning handles unlabeled data; complements supervised learning |",
        "| **2** | End-to-end ML project | Full workflow: data prep → models → evaluation → ensembling on credit fraud |",
        "",
        "**Tools:** scikit-learn, XGBoost (Extreme Gradient Boosting), LightGBM, Pandas, Jupyter",
        "",
        "**Core idea:** Unsupervised methods help when labels are scarce, features are high-dimensional, or you need anomaly scores and segments.",
    ))

    cells.append(md(
        "---",
        "",
        "## Part II — Unsupervised Learning Using Scikit-Learn (Chapters 3–6)",
        "",
        "| Chapter | Topic | Algorithms |",
        "|---------|-------|------------|",
        "| **3** | Dimensionality reduction | PCA (Principal Component Analysis), random projection, MNIST visualization |",
        "| **4** | Anomaly detection | PCA, ICA (Independent Component Analysis), dictionary learning + reconstruction error |",
        "| **5** | Clustering | k-Means, hierarchical clustering, DBSCAN, HDBSCAN on MNIST |",
        "| **6** | Group segmentation | k-Means, hierarchical, HDBSCAN on Lending Club loan data |",
        "",
        "**When to use scikit-learn:** Tabular data, classical algorithms, fast prototyping, interpretable pipelines on CPU.",
    ))

    cells.append(md(
        "---",
        "",
        "## Part III — Unsupervised Learning Using TensorFlow and Keras (Chapters 7–9)",
        "",
        "| Chapter | Topic | Key takeaway |",
        "|---------|-------|--------------|",
        "| **7** | Autoencoder concepts | Encoder-decoder, under/overcomplete, sparse, denoising, VAE (Variational Autoencoder) |",
        "| **8** | Hands-on autoencoder | Build and compare architectures on MNIST with Keras |",
        "| **9** | Semisupervised learning | Combine unsupervised features with supervised classifiers |",
        "",
        "**When to use TensorFlow/Keras:** Neural networks, image/audio data, GPU training, custom architectures, deep representation learning.",
    ))

    cells.append(md(
        "---",
        "",
        "## Part IV — Deep Unsupervised Learning (Chapters 10–13)",
        "",
        "| Chapter | Topic | Key takeaway |",
        "|---------|-------|--------------|",
        "| **10** | Recommender systems + RBMs | Restricted Boltzmann Machines for collaborative filtering (MovieLens) |",
        "| **11** | Deep Belief Networks | Stack RBMs; unsupervised pretraining improves classifiers |",
        "| **12** | GANs | Generator vs discriminator; DCGAN synthetic image generation |",
        "| **13** | Time series clustering | k-Shape, Euclidean k-Means, HDBSCAN on ECG data |",
        "",
        "**Bridge:** Part IV connects deep generative models (RBMs, DBNs, GANs) with specialized clustering (time series).",
    ))

    cells.append(md(
        "---",
        "",
        "## Key Tools — scikit-learn vs TensorFlow / Keras",
        "",
        "| Criterion | scikit-learn | TensorFlow / Keras |",
        "|-----------|--------------|-------------------|",
        "| **Best data types** | Tabular, small/medium numeric features | Images, sequences, large-scale deep nets |",
        "| **Algorithms** | PCA, clustering, classical ML | Autoencoders, GANs, custom neural nets |",
        "| **Training speed** | Fast on CPU for modest data | Benefits from GPU for deep models |",
        "| **Interpretability** | Often higher (components, centroids) | Lower — learned representations |",
        "| **Production** | sklearn Pipeline + joblib | TensorFlow SavedModel, TF Serving |",
        "| **Course chapters** | 1–6, parts of 2, 9, 11, 13 | 7–12 |",
        "",
        "**Practical rule:** Start with scikit-learn for exploration; move to Keras when linear models and PCA are not enough.",
    ))

    cells.append(md(
        "---",
        "",
        "## Course-Wide Summary Diagram",
        "",
        "```",
        "Hands-On Unsupervised Learning — Course Map",
        "│",
        "├── Part I: Fundamentals (Ch 1–2)",
        "│   └── Why unsupervised? End-to-end supervised workflow baseline",
        "│",
        "├── Part II: scikit-learn (Ch 3–6)",
        "│   ├── Dimensionality reduction (PCA, projection)",
        "│   ├── Anomaly detection (reconstruction error)",
        "│   └── Clustering & segmentation (k-Means, DBSCAN, HDBSCAN)",
        "│",
        "├── Part III: TensorFlow / Keras (Ch 7–9)",
        "│   ├── Autoencoders (compression, denoising)",
        "│   └── Semisupervised learning (unsupervised + supervised)",
        "│",
        "└── Part IV: Deep unsupervised (Ch 10–13)",
        "    ├── RBMs & recommenders (Ch 10)",
        "    ├── DBNs — stacked pretraining (Ch 11)",
        "    ├── GANs — adversarial generation (Ch 12)",
        "    └── Time series clustering — k-Shape (Ch 13)",
        "",
        "Unlabeled data → structure, features, anomalies, segments, or synthetic samples",
        "                              ↓",
        "              Improved supervised models & business insights",
        "```",
    ))

    cells.append(md(
        "---",
        "",
        "## Glossary — Course-Wide Terms",
        "",
        "| Term | One-line definition |",
        "|------|---------------------|",
        "| **Unsupervised learning** | Learn patterns from data without label guidance |",
        "| **Supervised learning** | Learn input→label mapping from labeled examples |",
        "| **Semisupervised learning** | Combine small labeled set with large unlabeled set |",
        "| **PCA (Principal Component Analysis)** | Linear dimensionality reduction maximizing variance |",
        "| **Anomaly detection** | Flag observations that deviate from normal patterns |",
        "| **Reconstruction error** | Difference between original and model-reconstructed input |",
        "| **Clustering** | Group similar points without predefined classes |",
        "| **k-Means** | Partition data into k centroid-based clusters |",
        "| **DBSCAN / HDBSCAN** | Density-based clustering; HDBSCAN is hierarchical variant |",
        "| **Autoencoder** | Neural network that compresses then reconstructs input |",
        "| **RBM (Restricted Boltzmann Machine)** | Two-layer generative energy-based model |",
        "| **DBN (Deep Belief Network)** | Stack of greedily pretrained RBMs |",
        "| **GAN (Generative Adversarial Network)** | Generator vs discriminator adversarial training |",
        "| **DCGAN** | Convolutional GAN architecture for images |",
        "| **k-Shape** | Time series clustering with shape-based distance |",
        "| **Feature engineering** | Create informative inputs for downstream models |",
        "| **Curse of dimensionality** | High-dimensional spaces make distance and density unreliable |",
        "| **Data drift** | Distribution shift over time requiring model adaptation |",
    ))

    cells.append(md(
        "---",
        "",
        "## Chapter 14 Summary",
        "",
        "You completed a journey from **fundamentals** through **classical scikit-learn**, **neural autoencoders**, and **deep generative models** to **specialized time series clustering**.",
        "",
        "**Three habits to keep:**",
        "",
        "1. **Always ask:** Would labels help evaluation even if training is unsupervised?",
        "2. **Match tool to problem:** Tabular → sklearn; images/deep reps → Keras; time series → tslearn.",
        "3. **Validate unsupervised output** with domain experts — clusters and anomalies need human interpretation.",
        "",
        "---",
        "",
        "### Practice reflection questions",
        "",
        "1. Which chapter's technique is most relevant to a problem you face at work or in research?",
        "2. Describe a project where unsupervised pretraining could reduce labeling cost.",
        "3. When would you trust anomaly scores without any labeled fraud examples?",
        "4. How would you explain the difference between PCA and an autoencoder to a non-technical stakeholder?",
        "5. What ethical risks exist when deploying GAN-generated synthetic data?",
        "6. Design a mini-project combining clustering (Part II) with autoencoder features (Part III).",
        "",
        "---",
        "",
        "**Congratulations — you have finished the Hands-On Unsupervised Learning course notebooks!**",
    ))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11.0",
                "mimetype": "text/x-python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
