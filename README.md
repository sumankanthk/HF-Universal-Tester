# 🤗 Universal Hugging Face Model Tester

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Inference_API-FFD21E)
![License](https://img.shields.io/badge/License-MIT-green)

A dynamic Streamlit web application that allows you to instantly test thousands of machine learning models hosted on the Hugging Face Hub without downloading them locally. 

By utilizing the **Hugging Face Serverless Inference API**, this app dynamically detects the modality of the requested model (e.g., text generation, image classification) and automatically renders the appropriate user interface.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hf-universal-tester.streamlit.app/)

## ✨ Features

* **Dynamic UI Routing:** Enter a Model ID, and the app fetches the model's metadata to automatically generate the correct interface (chat box, image uploader, etc.).
* **Serverless Inference:** Zero local GPU required. Model inference runs entirely on Hugging Face's shared API infrastructure.
* **Cold-Start Handling:** Includes robust error handling and loading states for models that need time to load into memory on the Hub.
* **Secure Credential Management:** Implements best practices for API key security using Streamlit Secrets.

## 🛠️ Supported Modalities

The app currently supports out-of-the-box routing for the following Hugging Face Pipeline tags:
* `text-generation` & `text2text-generation` (LLMs)
* `text-to-image` (Diffusion models)
* `image-classification` (Vision models)
* `summarization` (NLP models)

*(Easily extensible to support Audio, Video, and Object Detection by adding new routing blocks in `app.py`).*

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [[https://github.com/sumankanthk/HF-Universal-Tester](https://github.com/sumankanthk/HF-Universal-Tester).git]
cd hf-universal-tester
