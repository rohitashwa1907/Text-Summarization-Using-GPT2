import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import GPT2Tokenizer


    
""" Returns GPT2 tokenizer after adding special tokens """ 
def add_special_tokens(gpt2path):
	tokenizer = GPT2Tokenizer.from_pretrained(gpt2path)
	special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
	tokenizer.add_special_tokens(special_tokens)
	return tokenizer



def DataProcessor(tokenizer, datapath):
    
    data_csv = pd.read_csv(datapath, encoding= 'latin')
    
    """train valid split (90:10)"""
    traindata, validdata = train_test_split(data_csv, test_size= 0.10)
    
    
    """calculating length of all the article and summeries.
       As GPT2 supports only 1024 tokens at a time, we check for the 
       any items exceeding the length"""
       
    def len_calc(cell):
      try:
        return int(len(tokenizer(cell)['input_ids']))
      except:
        return np.nan
    
    """ calculating the length"""
    traindata.loc[:,'article_len'] = traindata.loc[:,'article'].apply(len_calc)
    traindata.loc[:,'summary_len'] = traindata.loc[:,'summary'].apply(len_calc)
    traindata.dropna(inplace=True)
    validdata.loc[:,'article_len'] = validdata.loc[:,'article'].apply(len_calc)
    validdata.loc[:,'summary_len'] = validdata.loc[:,'summary'].apply(len_calc)
    validdata.dropna(inplace=True)
    
    """ removing items exceeding the max length limit"""
    traindata.loc[:,'length'] = traindata.loc[:,'article_len'] + traindata.loc[:,'summary_len']
    traindata = traindata.loc[traindata.loc[:,'length'] <= 1018]
    traindata.reset_index(drop=True, inplace=True)
    validdata.loc[:,'length'] = validdata.loc[:,'article_len'] + validdata.loc[:,'summary_len']
    validdata = validdata.loc[validdata.loc[:,'length'] <= 1018]
    validdata.reset_index(drop=True, inplace=True)
    
    return traindata, validdata
        
        
       
	

