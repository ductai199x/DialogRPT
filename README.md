# Dialog Response Ranking with Reddit Data

_DSCI 691: Natural Language Processing with Deep Learning_

_Drexel University_

_06/10/2022_

-   Group member 1
    -   Name: Xi Chen
    -   Email: xc98@drexel.edu
-   Group member 2
    -   Name: Tai Nguyen
    -   Email: tdn47@drexel.edu
-   Group member 3
    -   Name: Tien Nguyen
    -   Email: thn44@drexel.edu
-   Group member 4
    -   Name: Raymond Yung
    -   Email: raymond.yung@drexel.edu

## Table of Contents

- [Dialog Response Ranking with Reddit Data](#dialog-response-ranking-with-reddit-data)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluating](#evaluating)
    - [Predicting](#predicting)
  - [Data](#data)
  - [Results and Discussion](#results-and-discussion)
  - [Limitations/Challenges](#limitationschallenges)

## Project Description

The goal of this project is to build dialog system evaluation models which can use measurements of engagement to rank dialog responses based on. We hope to use the deep learning architectures that we learned in DSCI691 to solve this problem. This repo provides a Pytorch implementation of the models.

Our project falls in the realm of **dialog system evaluation** as we are trying to predict how likely a dialog response is to elicit a positive reaction from the interlocutor. The models will be trained with Reddit threads and comments, considering feedback metrics including the number of replies on a post and the number of upvotes/downvotes.

The project is inspired by the dialog response ranking models proposed by [Microsoft Research NLP Group](https://github.com/iamxichen/DialogRPT) trained on 100+ millions of human feedback data. It can be used to create more engaging dialog agents by re-ranking the generated response candidates.

## Requirements

-   [Python](https://www.python.org/) – version 3.9.x
-   [Virtualenv](https://virtualenv.pypa.io/en/latest/)

## Installation

1. Clone the repository from GitHub:
    ```
    git clone https://github.com/ductai199x/DialogRPT
    cd DialogRPT
    ```
2. Use `virtualenv` to create a python virtual environment:
    ```
    virtualenv . --python=python3.9
    source bin/activate
    ```
3. Install all the pip packages dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Download GPT-2 pretrained + finetuned weights:
    ```
    python src/shared.py
    ```
5. Use `gdown` to download the checkpoints for our trained models
    ```
    gdown --fuzzy https://drive.google.com/file/d/1oKO5jIIXr69paRkFlrrwnkSyYLeigaga/view\?usp\=sharing
    unzip lightning_checkpoints.zip
    ```
6. Download the training data
    ```
    gdown --fuzzy https://drive.google.com/file/d/15CZUj1OlEPga8vmAq8dL0StMYDT-tAGF/view\?usp\=sharing
    unzip training_data.zip
    ```
7. Download the testing/evaluating data
    ```
    wget https://xiagnlp2.blob.core.windows.net/dialogrpt/test.zip 
    unzip test.zip -d data
    ```

## Usage

The entry of our code will be available at `src/main.py`. To get the usage, from the root of the project:
```
python src/main.py -h
```
You will see:
```
usage: main.py [-h] {train,eval,predict} ...

positional arguments:
  {train,eval,predict}
    train               Train existing architectures.
    eval                Evaluate existing architectures.
    predict             Predict using existing architectures.

optional arguments:
  -h, --help            show this help message and exit
```
To view the available architecture for each functionality (`train,eval,predict`), run `python src/main.py <function> -h`.

### Training
For example, to train the fully connected model on the updown set, run:
```
python src/main.py train --arch "FC-GPT" --feedback "updown"
```

### Evaluating

For example, to evaluate the fully connected model on the updown set, run:
```
python src/main.py eval --arch "FC-GPT" --feedback "updown"
```

### Predicting
For example, to predict with the context of "Can we restart 2020?", seq1 of "I think we should go back to the beginning, and start from the beginning." and seq2 of "I think so, yes.", run :
```
python src/main.py predict --arch "RPT" --feedback "updown" --context="Can we restart 2020?" --seq1="I think we should go back to the beginning, and start from the beginning." --seq2="I think so, yes."
```

## Data

Training dataset can be built with [this script](https://github.com/ductai199x/DialogRPT/blob/master/data.sh), which downloads data from a [third party dump](https://files.pushshift.io/reddit/). This will download ~44 G of compressed data. The script also builds pairs of comments for classification tasks using the data.py module.

Testing data can be downloaded using the command in the #7 of the [Installation](#installation) section


## Results and Discussion

**NOTE**: *Due to the limited storage and limited time, we are unable to upload the fullyconnected-glove models to the shared directory. Please contact us directly for further information.*

The pairwise accuracy and Spearman correlation scores on 5,000 test samples are listed in the tables below.

**Baseline Models**
| Feedback | Method                  | Pairwise Acc. | Spearman $\rho$ |
| -------- | ----------------------- | ------------: | --------------: |
| Width    | DialogRPT not-finetuned |        0.5146 |          0.0036 |
|          | DialogRPT               |        0.7581 |          0.4247 |
| Depth    | DialogRPT not-finetuned |        0.4962 |         -0.0012 |
|          | DialogRPT               |        0.6893 |          0.3159 |
| Updown   | DialogRPT not-finetuned |        0.5059 |         -0.0018 |
|          | DialogRPT               |        0.6808 |          0.2619 |

**Our Models**
| Feedback | Method                               | Pairwise Acc. | Spearman $\rho$ |
| -------- | ------------------------------------ | ------------: | --------------: |
| Width    | FullyConnected with GloVe Embeddings |        0.5000 |          0.1937 |
|          | FullyConnected with GPT-2 Embeddings |        0.6568 |          0.1900 |
|          | CNN with GPT-2 Embeddings            |        0.6653 |          0.2170 |
|          | LSTM with GPT-2 Embeddings           |        0.6502 |          0.1942 |
| Depth    | FullyConnected with GloVe Embeddings |        0.3667 |         -0.0864 |
|          | FullyConnected with GPT-2 Embeddings |        0.6673 |          0.1957 |
|          | CNN with GPT-2 Embeddings            |        0.6070 |          0.1410 |
|          | LSTM with GPT-2 Embeddings           |        0.5969 |          0.1285 |
| Updown   | FullyConnected with GloVe Embeddings |        0.5444 |          0.0532 |
|          | FullyConnected with GPT-2 Embeddings |        0.6122 |          0.1043 |
|          | CNN with GPT-2 Embeddings            |        0.5972 |          0.0921 |
|          | LSTM with GPT-2 Embeddings           |        0.5648 |          0.0573 |




1. **We built smaller-footprint models for the same task.** The purpose of the DialogRPT model is to evaluate the responses generated by dialog generation models. DialogRPT is built using GPT-2, which is very large with 400M parameters. Therefore, while the model performs well, it might not necessarily be efficient enough for direct deployment in the real world due to reasons such as high cost. There is also a very real carbon footprint concern, where [training a deep learning model can emit as much as five cars](https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/). Therefore, in this project, we tried to build more efficient, smaller-footprint models for the same task while achieving comparable quality.
  
2. **Fine-tuning significantly improved the performance of DialogRPT.** DialogRPT is fine-tuned based on GPT-2 and the fine-tuned models performed significantly better. The fine-tuning was done by applying initialized [DialoGPT medium model weights](https://github.com/microsoft/DialoGPT).

3. **We could not reproduce the level of performance of the DialogRPT models.** The DialogRPT models are based on the [GPT-2 architecture](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe). GPT-2 is a large scale transformer-based language model with some advanced learning concepts like Masked Self Attention, Multiple Heads, Residual Connections, Layer Normalization, etc., making it one of the best text generators out there. Our models are much simpler and lack these features.
   
4. **GPT-2 Embeddings outperform GloVe.** We built models based on the FullyConnected architecture from Assignment 4 of this class using both GPT-2 vs GloVe embeddings. GPT-2 embeddings clearly outperform GloVe embeddings because GloVe is static and has a fixed context-independent representation for each word, whereas GPT-2 embeddings are contextualized.  

5. **Dialog response ranking models can be fine-tuned on misuse.** Dialog response ranking models, like dialog response generation models, have the potential to be fine-tuned for misuse. Like how the [GPT-2 models can be used by extremist groups to generate synthetic propaganda for dangerous ideologies](https://openai.com/blog/gpt-2-1-5b-release/), our models can be trained to predict highly down-voted responses and be used in malicious applications.


## Limitations/Challenges
-   **Large dataset**: The uncompressed training data is over 100 G in size which requires large storage to store and handle the dataset. 
-   **Limited resource**: We used Google Cloud and AWS virtual machines to train our baselines and models due to the large dataset size. Unfortunately, we were not provided GPU with these VM which results in longer training time than what we expected. Each model and baseline was required to train with three different tasks (up-down, width, and depth) which required a lot of time for the training and evaluating process. We did not have VMs with GPU until this week.
-   **Inaccessible data**: We first followed the instruction from the reference paper to download the dataset. However, the instruction was not up to date and failed to achieve the entire dataset. We created our own data pipeline to introduce multiprocessing in order to run the data pipeline for three different years.
-   **Data preprocessing**: We also had to write dataloader.py to speed up the data loading process. Our dataloader.py can load three times as fast as pandas.read_csv due to efficient multiprocessing and prefetching.
-   **Limited time**: We had only four weeks to work on the project including generating ideas, understanding datasets, building up pipelines, training and evaluating baselines and models, and completing assignments 4 and 5. If we had more time, we would try to improve our pairwise accuracy and spearman correlation.
- Finally, most of the members were new to NLP and deep learning. It took us a little while to fully understand contrastive learning and the DialogRPT model which is significant for our project. However, we worked closely together and were able to finish the project given the limited amount of time. 