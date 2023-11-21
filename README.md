<div style="background-color: #000000;">
    <img src="https://raw.githubusercontent.com/stevienovak/garage/main/Web%201920%20–%201.jpg" alt="AlphaPredict_logo" width="1000">
</div>



<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build">
  <img src="https://img.shields.io/badge/docs-failing-red" alt="Documentation">
  <img src="https://img.shields.io/badge/pypi-v0.3.3-blue" alt="PyPI Version">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/downloads-100%2Fmonth-brightgreen" alt="Downloads">
  <img src="https://img.shields.io/badge/pytorch-1.4%2B-orange" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/python-3.7%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/BERT%2B-purple" alt="BERT Version">
  <img src="https://img.shields.io/badge/FINBERT%2B-indigo" alt="FINBERT Version">
</p>


# SentimentDrivenRLTrader

Reinforcement Learning Framework for Stock Trading Using Sentiment Analysis. This repository hosts a novel RL-based trading model that integrates real-time market sentiment to make informed buy and sell decisions. 

## Downloading Data
Before running the notebook, you'll first need to download all data we'll be using. This data is located in the models.tar.gz and training_data.tar.gz tarballs. As always, the first step is to clone the repository.


<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">git clone https://github.com/stevienovak/SentimentDrivenRLTrader.git</code></pre>
</div>

Next, we will navigate to the newly created directory and run the following commands.

<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">tar -xvzf base_models.tar.gz</code></pre>
</div>

<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">tar -xvzf dataset.tar.gz</code></pre>
</div>

## Requirements and Installation
In order to run the notebooks, you'll need the following libraries:

- [TensorFLow](https://www.tensorflow.org/install/)

## Installing Anaconda Python and TensorFlow

The easiest way to install TensorFlow as well as NumPy, Jupyter, and matplotlib is to start with the Anaconda Python distribution.
Follow the installation instructions for [Anaconda Python](https://www.anaconda.com/download). We recommend using Python 3.7.

Follow the platform-specific [TensorFlow](https://www.tensorflow.org/install/) installation instructions. Be sure to follow the "Installing with Anaconda" process, and create a Conda environment named tensorflow.

If you aren't still inside your Conda TensorFlow environment, enter it by opening your terminal and typing
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">source activate tensorflow</code></pre>
</div>

Use <div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">cd</code></pre>
</div> to navigate into the top directory of the repo on your machine
Launch Jupyter by entering
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">jupyter lab</code></pre>
</div>



