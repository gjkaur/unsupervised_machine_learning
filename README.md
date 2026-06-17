# Hands-On Unsupervised Learning Using Python

**Subtitle:** How to Build Applied Machine Learning Solutions from Unlabeled Data  
**Author:** Ankur A. Patel  
**Publisher:** O'Reilly Media  
**Published:** March 2019 | **~359 pages**

---

## Table of Contents

### Preface

- A Brief History of Machine Learning
- AI Is Back, but Why Now?
- The Emergence of Applied AI
- Major Milestones in Applied AI over the Past 20 Years
- From Narrow AI to AGI
- Objective and Approach
- Prerequisites
- Roadmap
- Conventions Used in This Book
- Using Code Examples
- O'Reilly Online Learning
- How to Contact Us

---

## Part I — Fundamentals of Unsupervised Learning

### Chapter 1 — Unsupervised Learning in the Machine Learning Ecosystem

**Supervised vs. Unsupervised**

- The Strengths and Weaknesses of Supervised Learning
- The Strengths and Weaknesses of Unsupervised Learning

**Using Unsupervised Learning to Improve Machine Learning Solutions**

- Insufficient labeled data
- Overfitting
- Curse of dimensionality
- Feature engineering
- Outliers
- Data drift

**A Closer Look at Supervised Algorithms**

- *Linear Methods:* Linear regression, Logistic regression
- *Neighborhood-Based Methods:* k-nearest neighbors
- *Tree-Based Methods:* Single decision tree, Bagging, Random forests, Boosting

**A Closer Look at Unsupervised Algorithms**

- *Dimensionality Reduction*
  - Linear projection: PCA, SVD, Random projection
  - Manifold learning: Isomap, t-SNE, Dictionary learning
- *Clustering:* k-means, Hierarchical clustering, DBSCAN
- *Feature Extraction:* Autoencoders; feature extraction via supervised feedforward networks
- *Unsupervised Deep Learning:* Unsupervised pretraining, RBMs, DBNs, GANs

**Successful Applications of Unsupervised Learning**

- Anomaly detection
- Group segmentation

---

### Chapter 2 — End-to-End Machine Learning Project

**Environment Setup**

- Version Control: Git
- Clone the Hands-On Unsupervised Learning Git Repository
- Scientific Libraries: Anaconda Distribution of Python
- Neural Networks: TensorFlow and Keras
- Gradient Boosting: XGBoost and LightGBM
- Clustering Algorithms
- Interactive Computing Environment: Jupyter Notebook

**Data Preparation**

- *Data Acquisition:* Download data, import libraries, read/preview data
- *Data Exploration:* Summary statistics, nonnumerical values, distinct values
- *Feature Matrix & Labels:* Create X and Y, standardize X
- *Feature Engineering & Selection:* Check feature correlation

**Model Preparation**

- Train/test split
- Cost function selection
- k-fold cross-validation sets

**Machine Learning Models (Part I)**

- Model #1: Logistic Regression (hyperparameters, training, evaluation)
- Evaluation Metrics & ROC curve

**Machine Learning Models (Part II)**

- Model #2: Random Forests
- Model #3: Gradient Boosting (XGBoost)
- Model #4: Gradient Boosting (LightGBM)

**Evaluation of the Four Models Using the Test Set**

**Ensembles**

- Stacking (layer-one predictions, hyperparameters, training, evaluation)

---

## Part II — Unsupervised Learning Using Scikit-Learn

### Chapter 3 — Dimensionality Reduction

**The Motivation for Dimensionality Reduction**

**The MNIST Digits Database**

- Data acquisition and exploration
- Load datasets, verify shapes, create DataFrames, explore data, display images

**Dimensionality Reduction Algorithms**

- Linear Projection vs. Manifold Learning

**Principal Component Analysis (PCA)**

- Set hyperparameters, apply PCA, evaluate PCA, visualize separation in space

**Random Projection**

- Gaussian Random Projection
- Sparse Random Projection

---

### Chapter 4 — Anomaly Detection

**Credit Card Fraud Detection**

- Prepare the data
- Define anomaly score function
- Define evaluation metrics
- Define plotting function

**Normal PCA Anomaly Detection**

- PCA components equal to original dimensions
- Search for optimal number of principal components

**Fraud Detection on the Test Set**

- Normal PCA anomaly detection
- ICA anomaly detection
- Dictionary learning anomaly detection

---

### Chapter 5 — Clustering

**MNIST Digits Dataset — Data Preparation**

**k-Means**

- k-Means inertia
- Evaluating clustering results
- k-Means accuracy
- k-Means and number of principal components
- k-Means on the original dataset

**Hierarchical Clustering**

- Agglomerative hierarchical clustering
- The dendrogram
- Evaluating clustering results

**DBSCAN**

- DBSCAN algorithm
- Applying DBSCAN to the dataset
- HDBSCAN

---

### Chapter 6 — Group Segmentation

**Lending Club Data**

**Data Preparation**

- Load libraries
- Explore the data
- Transform string format to numerical format
- Impute missing values
- Engineer features
- Select final feature set and perform scaling

**Designate Labels for Evaluation Goodness of the Clusters**

**Clustering Applications**

- k-Means application
- Hierarchical clustering application
- HDBSCAN application

---

## Part III — Unsupervised Learning Using TensorFlow and Keras

### Chapter 7 — Autoencoders

**Neural Networks**

**TensorFlow**

- TensorFlow example

**Keras**

**Autoencoder Concepts**

- The encoder and the decoder
- Undercomplete autoencoders
- Overcomplete autoencoders
- Dense vs. sparse autoencoders
- Denoising autoencoder
- Variational autoencoder

---

### Chapter 8 — Hands-On Autoencoder

**Data Preparation**

**The Components of an Autoencoder**

- Activation functions

**Our First Autoencoder**

- Loss function
- Optimizer
- Training the model
- Evaluating on the test set

**Two-Layer Undercomplete Autoencoder with Linear Activation**

- Increasing the number of nodes
- Adding more hidden layers

**Additional Architectures**

- Nonlinear autoencoder
- Overcomplete autoencoder (linear activation)
- Overcomplete autoencoder with dropout
- Sparse overcomplete autoencoder (with/without dropout)

**Working with Noisy Datasets — Denoising Autoencoder**

- Two-layer denoising undercomplete autoencoder (linear activation)
- Two-layer denoising overcomplete autoencoder (linear activation)
- Two-layer denoising overcomplete autoencoder (ReLU activation)

---

### Chapter 9 — Semisupervised Learning

- Data preparation
- Supervised model
- Unsupervised model
- Semisupervised model
- The power of supervised and unsupervised learning

---

## Part IV — Deep Unsupervised Learning Using TensorFlow and Keras

### Chapter 10 — Recommender Systems Using Restricted Boltzmann Machines

**Boltzmann Machines**

- Restricted Boltzmann Machines (RBMs)

**Recommender Systems**

- Collaborative filtering
- The Netflix Prize

**MovieLens Dataset**

- Data preparation
- Cost function: Mean Squared Error
- Baseline experiments

**Matrix Factorization**

- One, three, and five latent factors

**Collaborative Filtering Using RBMs**

- RBM neural network architecture
- Build components of the RBM class
- Train RBM recommender system

---

### Chapter 11 — Feature Detection Using Deep Belief Networks

**Restricted Boltzmann Machines**

- Build components of the RBM class
- Generate images using the RBM model
- View intermediate feature detectors

**Train the Three RBMs for the DBN**

- Examine feature detectors
- View generated images

**The Full DBN**

- How training of a DBN works
- Train the DBN

**How Unsupervised Learning Helps Supervised Learning**

- Generate images to build a better image classifier

**Image Classifier Using LightGBM**

- Supervised-only approach
- Combined unsupervised + supervised solution

---

### Chapter 12 — Generative Adversarial Networks

**GANs — The Concept**

- The power of GANs

**DCGANs Revisited**

- Generator of the DCGAN
- Discriminator of the DCGAN
- Discriminator and adversarial models
- DCGAN for the MNIST dataset

**MNIST DCGAN in Action**

- Synthetic image generation

---

### Chapter 13 — Time Series Clustering

**Approach to Time Series Clustering**

- k-Shape

**Time Series Clustering Using k-Shape on ECGFiveDays**

- Data preparation
- Training and evaluation

**Time Series Clustering Using k-Shape on ECG5000**

- Data preparation
- Training and evaluation

**Comparing Time Series Clustering Algorithms**

- Full run with k-Shape
- Full run with k-Means
- Full run with HDBSCAN
- Comparing all three approaches

---

## Chapter 14 — Conclusion

**Unsupervised Learning**

- Scikit-Learn
- TensorFlow and Keras

---

## Summary by Topic Area

| Topic Area | Chapters | Key Tools |
|---|---|---|
| ML fundamentals & workflow | 1–2 | scikit-learn, XGBoost, LightGBM |
| Dimensionality reduction | 3 | PCA, random projection, MNIST |
| Anomaly detection | 4 | PCA, ICA, dictionary learning |
| Clustering | 5–6, 13 | k-means, hierarchical, DBSCAN, HDBSCAN, k-Shape |
| Autoencoders | 7–8 | TensorFlow, Keras |
| Semisupervised learning | 9 | Combined supervised + unsupervised |
| Recommender systems | 10 | RBMs, MovieLens |
| Deep belief networks | 11 | RBMs, DBNs, LightGBM |
| Generative models | 12 | DCGANs, MNIST |
| Time series clustering | 13 | ECG datasets |

---

## Resources

- [Official code repository](https://github.com/aapatel09/handson-unsupervised-learning)
- [O'Reilly book page](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/)
