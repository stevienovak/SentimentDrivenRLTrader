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

## Guidelines
In this section, I briefly explain different parts of the project and how to change each. The data for the project downloaded from Yahoo Finance where you can search for a specific market there and download your data under the Historical Data section. Then you create a directory with the name of the stock under the data directory and put the .csv file there. Make sure the dataframe is in the format as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/basic_dataframe.jpg" alt="Basic_df" height="300">
</p>

There are two models you can obtain from this repository
- Model 1: our unique pretrained FinBERT Model trained on 18 months ( 01 Jan 2022 to 30 Jun 2023) of Stockwit data.
- Model 2: our RL model. []

To load the FinBERT Model on you local environment via the following code block (please note that we are using google colab here, code block as follows: 

<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">from torch.utils.data import SequentialSampler</code></pre>
</div>
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">model = BertForSequenceClassification.from_pretrained("/content/drive/<local_drive_where_model_was_saved></code></pre>
</div>
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')></code></pre>
</div>
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">device = torch.device("cuda" if torch.cuda.is_available() else "cpu")></code></pre>
</div>
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">model.to(device)</code></pre>
</div>

Think yourself as a savy investor, you can test out your newly model with your comments as follows!!: 
<p align="center">
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/BERT_model_test.jpg" alt="w_indicators" height="300">
</p>


you can add on technical indicators by performing the following: 
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">!pip install finta</code></pre>
</div>
followed by:
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">from finta import TA</code></pre>
</div>
Read more about the documentation on finta here: 

- [Finta](https://www.tensorflow.org/install/](https://github.com/peerchemist/finta/tree/master)https://github.com/peerchemist/finta/tree/master)

This will give you a df as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/DF_with_indicators.jpg" alt="w_indicators" height="300">
</p>

alternatively, add on the sentiment score given by our FinBert model, this will provide you a df as follows: 

<p align="center">
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/df_w_sentiments.jpg" alt="w_sentiment" height="300">
</p>




