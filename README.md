# ¡Hola Transformer! 🤖✨

**Building an English-to-Spanish Translator from Scratch with JAX, Flax (NNX), Grain, Optax, and Orbax**

---

Welcome, intrepid coder! Embark on an epic quest to build your very own Neural Machine Translation (NMT) model – a Transformer capable of translating English into Spanish. Forget dusty textbooks and overly complicated diagrams; we're building this sassy translator piece by piece using the cutting-edge JAX ecosystem!

This repository hosts the Google Colab notebook that serves as your guide. Everything you need, from setup to training and inference, is packed into this single, interactive document.

## 🚀 Dive Into the Tutorial

The entire tutorial and code are in this Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HI-iZyQQ4DTJKfL8AuM1KpKdkVhTlZtc)

Simply click the badge above or the link below to open it directly in Google Colab and start coding!

🔗 [https://colab.research.google.com/drive/1HI-iZyQQ4DTJKfL8AuM1KpKdkVhTlZtc](https://colab.research.google.com/drive/1HI-iZyQQ4DTJKfL8AuM1KpKdkVhTlZtc)

## ✨ What You'll Build & Learn

In this hands-on tutorial, you will:

* 🔧 Set up a modern AI development environment with **JAX** and **Flax (NNX API)**.
* 📦 Prepare an English-Spanish dataset and tokenize text using **SentencePiece**.
* 🚄 Build a high-performance data pipeline with **Grain** to efficiently feed your model.
* 🏗️ Construct the **Transformer architecture** from its fundamental components:
    * Positional & Token Embeddings
    * Multi-Head Attention (leveraging Flax's optimized implementation)
    * Position-wise Feed-Forward Networks
    * The full Encoder stack
    * The full Decoder stack (including masked self-attention and cross-attention)
* 🏋️ Configure and run a training loop using **Optax** for optimization and loss calculation.
* 💾 Implement reliable checkpointing with **Orbax** to save your model's progress.
* 🗣️ Build a **Sampler** for generating translations during inference, complete with KV caching optimization.
* 🔬 See your creation translate example sentences!

## 🛠️ Technologies Used

* **JAX**: For blazing-fast numerical computing.
* **Flax (NNX)**: Our flexible LEGO set for neural networks in JAX.
* **Grain**: The data pipeline powerhouse.
* **Optax**: JAX-compatible optimizers and loss functions.
* **Orbax**: Checkpointing made easy.
* **SentencePiece**: Sub-word text tokenization.
* **NumPy**: The classic numerical library.
* **tqdm**: For delightful progress bars.
* **Matplotlib**: For plotting training curves.

## 📊 Dataset & Tokenizer

* **Dataset:** English-Spanish sentence pairs (`spa-eng.zip`) from TensorFlow's public data. The notebook automatically downloads and extracts this.
* **Tokenizer:** A pre-trained SentencePiece model (`eng_spa_spm.model`) compatible with the dataset. The notebook also handles downloading this.

## 🏃‍♀️ Getting Started

The easiest way to run this project is directly in Google Colab using the link above. Colab provides the necessary environment, including free access to GPUs or TPUs. The notebook includes cells to install all required libraries.

If you prefer to run locally, you will need:
1.  Python 3.8+
2.  Install the required libraries (see the `!pip install` command in the notebook for the full list).
3.  A compatible JAX backend (CPU, GPU, or TPU).

## 💾 Checkpoints

The notebook is configured to download pre-trained checkpoint weights (for the 'big' configuration) from a GitHub URL. This allows you to potentially start from a partially trained model or examine the structure of the saved checkpoints using Orbax. The training loop also saves new checkpoints locally.

## 📝 Example Translations

After training and building the `Sampler`, you can input English sentences and see the generated Spanish output.

*(Note: Actual translation quality depends on the model size and duration of training run in the notebook.)*

## 🌱 Future Enhancements

This tutorial provides a solid foundation. Here are some ideas to take it further:

* Train the model for more epochs or with a larger configuration.
* Implement quantitative evaluation metrics like BLEU score using `metrax`.
* Explore advanced decoding strategies such as Beam Search.
* Experiment with different datasets or language pairs.
* Investigate distributed training for larger models.

---

¡Happy Coding! 🎉
