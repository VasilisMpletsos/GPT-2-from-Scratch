# GPT-2 From Scratch

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ff8f00?logo=tensorflow&logoColor=white)](#)

## 🚀 About This Project

This project follows Andrej Karpathy’s tutorial ([Youtube Link](https://www.youtube.com/watch?v=l8pRSuU81PU)) and the paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on building the GPT-2 architecture from scratch.
If you want to see the original implementation is the latest model published openly from Open-AI and it is available on [Hugging Face](https://huggingface.co/openai-community/gpt2) and also the code can be found at their [OpenAI GPT-2 Repo](https://github.com/openai/gpt-2/tree/master/src).

Below you can see the model architecture as reported from pytorch:
<img src="./assets/gpt2model.png" alt="GPT-2 Model" style="width:550px; border-radius:10px; border:2px solid #eee;" />

You can clearly observe the following:

- Vocabulary Size = 50257
- Context length = 1024
- Embedding Size = 768
- Output head = Embedding Size → Vocabulary Size
- And the targeted Layer Normalization placements as described inside the paper

![Layer Norm Placements](./assets/layer_norm_placements.png)
