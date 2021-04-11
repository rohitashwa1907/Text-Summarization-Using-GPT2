import torch
from torch.utils.data import Dataset


""" Preparing dataset for GPT-2"""


""" Preparing dataset for GPT-2"""
class GPT2Dataset(Dataset):
  
  def __init__(self, data, tokenizer):
    self.data = data
    self.tokenizer = tokenizer

  
  def __len__(self):
    return len(self.data)

  
  """ token type adds unique labels for twos separate sections. Here the artcle and target summary is two separate section """ 
  def token_type_labels(self, content, max_len=1024):
    item = content
    num_seg_a = item.index(self.tokenizer.additional_special_tokens_ids[1]) + 1
    end_index = item.index(self.tokenizer.eos_token_id)
    num_seg_b = end_index - num_seg_a + 1
    num_pad = max_len - (num_seg_a + num_seg_b)
    segment_ids = [self.tokenizer.additional_special_tokens_ids[0]]*num_seg_a + [self.tokenizer.additional_special_tokens_ids[1]]*num_seg_b + [self.tokenizer.pad_token_id]*num_pad
    return segment_ids

  
  """ it is used for calculating training loss. Loss is only calculated for the target summary
      and not on the article. All those with lable -100 will be ignored while calculating loss."""     
  def lm_labels(self, input_ids, type_labels):
    temp_list = []
    for token,segment in zip(input_ids,type_labels):
      if segment == self.tokenizer.additional_special_tokens_ids[1]:
        temp_list.append(token)
      else:
        temp_list.append(-100)
    return temp_list


  def __getitem__(self, idx):
    
    #article_id = self.tokenizer.encode(self.data.iloc[idx]['article'])
    #summary_id = self.tokenizer.encode(self.data.iloc[idx]['summary'])
    article = self.data.iloc[idx]['article']
    summary = self.data.iloc[idx]['summary']
    sample = self.tokenizer.encode(self.tokenizer.pad_token)*1024
    input_id = self.tokenizer.encode(text = '<|startoftext|> ' + article + ' <|summarize|> '+ summary + ' <|endoftext|>')
    attention_mask = torch.zeros((self.tokenizer.model_max_length), dtype= torch.int64)
    attention_mask[:len(input_id)] = 1
    #print(len(article_id), len(summary_id), len(input_id))
    sample[:len(input_id)] = input_id
    token_id = torch.tensor(self.token_type_labels(sample))
    lm_label = torch.tensor(self.lm_labels(sample, token_id))
    
    
    return torch.tensor(sample), attention_mask, token_id, lm_label