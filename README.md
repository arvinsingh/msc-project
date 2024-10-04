# Project Janus: 4D Facial Dynamics Analysis for Fluency Detection

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Experimentation](#Experimentation)
7. [Future Work](#future-work)


## Introduction

This repository contains code for a deep learning project focused on detecting fluency using a **bimodal architecture**. The model takes two inputs—audio waveforms and a sequence of interconnected 3D facial landmarks—and uses a combination of **LSTM** for audio and **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** for facial landmark dynamics.

The goal is to leverage multimodal data to improve fluency detection accuracy by modeling temporal and spatial aspects of speech and facial expression.

## Architecture

The model consists of two primary components:

- **LSTM** for processing audio waveform inputs.
- **ST-GCN** for processing temporal and spatial relationships between 3D facial landmarks.

These components are merged in a later stage for combined feature learning, and the final model is trained using a triplet loss function to handle the small sample size through few-shot learning.

## Dataset

The dataset consists of audio recordings and corresponding 3D facial landmark data. The dataset was extended in 2023 from its initial version in 2018, and much effort was made to clean and synchronize the data.

### Key challenges:
- Missing labels
- Corrupted files
- Noisy audio
- Unsynchronized audio and landmarks
- Ensuring correct labels and triplet generation

## Installation

### Prerequisites

1. **Python 3.8+**
2. **PyTorch** (preferably with CUDA for GPU support)
3. **Other dependencies**: See the `requirements.txt` file.

### Setup

1. Clone the repository - Currently in private mode:
   ```bash
   git clone git@github.com:arvinsingh/msc-project.git
   cd deep-learning-fluency-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. If you want to use a GPU, ensure that CUDA is properly installed and that PyTorch is configured to use it.

## Usage

### Dataset Preparation

- Place the dataset in the `data\preprocessed` folder. Ensure that the dataset includes both the **audio waveform files** and **3D facial landmark sequences**.
- Scripts in `src\data` are included to help with data cleaning, and synchronization between the two input types.

### Model Configuration

You can configure the model by editing the `src\config\config.py` file, where you can set parameters like batch size, learning rate, and the number of epochs.

## Experimentation

Check `notebooks\notebook.ipynb`

## Future Work

In the short and long term, the following features and improvements are planned:

- **1 month**: Incorporate synthetic talking head generation for automatic landmark annotation and further clean noisy audio data.
- **6 months**: Develop an app for wider user testing, implement transfer learning techniques, and expand the dataset using faster data collection methods.
