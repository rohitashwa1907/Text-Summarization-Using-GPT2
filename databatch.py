import torch
from transformers import GPT2Tokenizer


""" Preparing data batch for GPT-2"""

""" We are setting batch size as 1 due to gpu memory limitation and will use gradient accmulation. 
Hence we will not pad the text in order to make he training procees more efficient"""

def smart_batching(data, tokenizer):

  def gpt_labels(input_id):

    num_seg_a = input_id.index(tokenizer.additional_special_tokens_ids[1]) + 1
    end_index = input_id.index(tokenizer.eos_token_id)
    num_seg_b = end_index - num_seg_a + 1
    type_id = [0]*num_seg_a + [1]*num_seg_b
    lm_label = [-100]*num_seg_a + input_id[num_seg_a :]
    return type_id, lm_label


  databatch = []
  for index, row in data.iterrows():

    article = row['article']
    summary = row['summary']
    input_id = tokenizer.encode(text = '<|startoftext|> ' + article + ' <|summarize|> '+ summary + ' <|endoftext|>')
    type_id, lm_label = torch.tensor(gpt_labels(input_id))
    input_id = torch.tensor(input_id)

    databatch.append((input_id, type_id, lm_label))

  return databatch