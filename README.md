<div style="background-color: #000000;">
    <img src="https://raw.githubusercontent.com/stevienovak/garage/main/Web%201920%20–%201.jpg" alt="AlphaPredict_logo" width="1000">
</div>



<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build">
  <img src="https://img.shields.io/badge/docs-red" alt="Documentation">
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

This repository hosts a novel reinforcement Learning ("RL") -based trading model that integrates market sentiment to make informed buy and sell decisions. 

Main features of this library are: 
- A suite of reinforcement learning models, comprising ten tailored for discrete action spaces and two designed for continuous action spaces. The discrete models are distributed evenly across Tripartite and Quintuple Sentiment Categories, while the continuous models are split equally between these two sentiment classification schemes.
- NLP FinBERT Model trained on Stockwit investors' comments data.
- Notebooks to help you in cleaning and preparing files with the pricing and sentiment score columns to be used for RL models.
- Extra features which will be useful for future research. 

## 1. Downloading Repository 
Before running the notebook, the first step is to clone the repository: 

<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">git clone https://github.com/stevienovak/SentimentDrivenRLTrader.git</code></pre>
</div>

## 2. Requirements and Installation
In order to run the notebooks, you'll need the following key libraries:

- [TensorFLow](https://www.tensorflow.org/install/)
- [pyTorch](https://pytorch.org/get-started/locally/)

### 2.1 Installing Anaconda Python and TensorFlow

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

### 2.2 Install the required modules
<div style="background-color: #0d1117; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
  <pre style="margin: 0;"><code style="color: #c9d1d9; background-color: #0d1117;">pip install -r requirements.txt</code></pre>
</div>



## 3. Guidelines & Project Structure
In this section, we briefly explain different parts of the project and how to change each. Our Workflow is as follows:

<p align="center">

  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/0712_Flow%20Chart.jpg" alt="Flow_Chart" height="400">
</p>

### 3.1 Data Access
For the project, we sourced our data from the following platforms, adhering to their respective data licensing terms and conditions:

- Yahoo Finance: Here, you can search for specific markets and download the necessary data from the 'Historical Data' section.
- Stocktwits: While we have utilized data from Stocktwits, we fully respect and comply with their data usage policies. Consequently, we will not display or detail the methods used to obtain the data. However, you can access investor sentiments and discussions directly at [Stocktwits](https://stocktwits.com).

Contained within the `00_Data folder` is a compact dataset, consisting of 100 entries, provided for you, the reader, to engage with our models. This sample is intended to help you understand the file architecture and the specific columns necessary to duplicate our research and results.

### 3.2 Data Cleaning and Pre-processing
You can find the necessary data cleaning and pre-processing procedures in the notebooks located in the `01_Data Cleaning, Sentiment Analysis` directory. These steps are crucial for refining, preprocessing, and balancing the data, ensuring it's primed for use with our Tripartite and Quintuple sentiment models.

## 4. Models 
There are two models you can obtain from this repository

### 4.1 Quinduple / Tripartite Sentiment Categories Model 
- You can obtian from `02_Bert implementation, Quinduple Sentiment Categories Genesis` our unique pretrained FinBERT Model trained and validated using 01 Jan 2009 to 31 December 2016 Stockwit data.


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

If you followed the steps so far and done the implementation properly, you will get the table as follows:

<p align="center">
<img src="https://raw.githubusercontent.com/stevienovak/garage/main/Quintuple_Tripartite_Sentiment_Categories.jpg"alt="w_indicators" height="300">
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
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/df_w_sentiments.jpg" alt="w_indicators" height="300">
</p>

Congrats on getting to this step! you can now proceed with using our reinforcement learning models. 

### 4.2 Reinforcement Learning Model 
- For discrete action spaces, we used Deep Q-Network (DQN), Double Deep Q-Network (DDQN), Distributional Reinforcement Learning (C51), and Quantile Regression-based Distributional RL (QRDQN).
- Proximal Policy Optimization (PPO) was employed for continuous action spaces.

You may access them from the following folders: `04_RL Models - Quinduple Sentiment Categories` and `05_RL Models - Tripartite Sentiment Categories`. 

## 5. Outputs
Using the code in the Continuous action space `036_PPO.ipynb` and `046_PPO.ipynb`, you can plot the results as follows:
<p align="center">
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/PPO%20Model.jpg" alt="w_sentiment" height="300">
</p>

## 6. Outputs
Benchmark your model's performance against that of mutual funds using our Python package provided in the `06_Benchmark Against Mutual Funds` directory.

## 7. Possible Approaches to Future Work
When working on the project, we developed a function to calculate 'novelty' and 'volume' metrics from stock data, aiming to explore deeper insights into market dynamics. The function iterates through our stock data, calculating these metrics for different time windows (0.5, 1, 3, 5, and 7 days). You can retrieve the function from the following folder `07_Useful features for future reserarch`. 

While these features offered potential insights into market trends and stock behavior, we ultimately decided not to include them in our final models to prevent overfitting and keeping model simplicity. we conducted a correlation analysis to explore the relationship between the novelty and volume features and the five sentiment categories. 

<p align="center">
  <img src="https://raw.githubusercontent.com/stevienovak/garage/main/Correlation_Analysis.jpg" alt="w_sentiment" height="1000">
</p>

Our analysis revealed no significant correlation between these newly developed features and the sentiment categories. Nonetheless, it's important to note that the absence of a strong correlation doesn't necessarily render these features irrelevant for our model. Even a modest correlation score might contribute to enhancing the performance of the reinforcement learning model. 

Although we chose not to incorporate these features in our current RL model, they present a promising avenue for future research, offering potential insights that could further refine algorithmic strategies!


