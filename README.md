# Description
The objective of this project fine-tune the pre-trained Transformer Decoder-based language GPT2 models to obtain a very powerful abstractive text summarizer.

## setting up the environment
install from the requirements.txt

`pip install -r requirements.txt`

## Training the GPT2

`mkdir fine_tuned_folder`

`python train_command_line.py --epochs=1 --data_path='insert-path-to-training-data-here' --model_arch_name='name-of-the-gpt2-model' --model_directory='fine_tuned_folder'`

## Generating Summaries

`python eval.py --input_file='insert-path-to-text-data-here' --model_directory='insert-path-to-pretained-model-here'  --model_arch_name='name-of-the-gpt2-model' --num_of_samples='num-of-samples-to-generate`

## Notebook

This folder contains colab notebook that guide you through the summarization by GPT-2. You should be able to [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitashwa1907/Text-Summarization-Using-GPT2/blob/main/Notebook/GPT2_Summarizer.ipynb)
, and play with your data. 

