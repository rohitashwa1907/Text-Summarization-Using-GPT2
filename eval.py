import os
import torch
import argparse
import warnings
import helper as hlp
from transformers import GPT2LMHeadModel


def eval(args):
    
    warnings.filterwarnings("ignore")
    
    """ set the device """
    if torch.cuda.is_available():
      device = torch.device('cuda:0')
    else:
      device = torch.device('cpu')
      
   
    """ tokenizer for gpt2 model"""
    tokenizer = hlp.add_special_tokens(args.model_arch_name)
    
    
    """ downloading the gpt2 model using huggingface"""
    print('DOWNLOADING MODEL FROM HUGGINGFACE \U0001F917 \U0001F917........................................................')
    model = GPT2LMHeadModel.from_pretrained(args.model_arch_name)
    model.resize_token_embeddings(len(tokenizer))
    
    
    
    """ loading previously saved model"""
    if device.type == 'cuda':
        model.load_state_dict(torch.load(args.model_directory))
    elif device.type == 'cpu':
        model.load_state_dict(torch.load(args.model_directory, map_location = torch.device('cpu')))
    model.to(device)
    
    """ read the text file"""
    file1 = open(args.input_file,'r')
    input_text = file1.read()
    file1.close()
    
    sample_article = "<|startoftext|> " + input_text + " <|summarize|>"
    
    """ checking the length of the input text """
    inp_length= int(len(tokenizer(sample_article)['input_ids']))
    
    if inp_length < 1018:    
        GPT2_input = torch.tensor(tokenizer.encode(sample_article), dtype=torch.long)
        input_id = GPT2_input.to(device)
        
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        
        # set top_k to 50
        sample_output = model.generate(
            input_ids = input_id.unsqueeze(0), 
            temperature = 1,
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            decoder_start_token_id= '<|summarize|>',
            do_sample=True, 
            max_length=200 + len(input_id),
            min_length=20 + len(input_id),
            top_p = 0.8, 
            top_k=50,
            no_repeat_ngram_size=3,
            num_return_sequences= args.num_of_samples
        )
        
        print("HERE ARE SOME SUMMARIES TO TRY FROM :\U0001F607	 \U0001F607	 ")
        for i in range(len(sample_output)):
            print("\n" + 100 * '-')
            print(tokenizer.decode(sample_output[i, len(input_id):], skip_special_tokens=True, clean_up_tokenization_spaces =True))
            
    else:
        print('Sorry!! \U0001F641 \U0001F641 Input is too long for me. Please let me try a smaller one.')
    




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='GPT2 Evaluator')
    
    parser.add_argument('--input_file', type=str, help='provide the path to the input file (.txt file)')
    
    parser.add_argument('--num_of_samples', type=int, default = 3,
                        help='number of summary samples (default: 3)')
    
    parser.add_argument('--model_directory', type=str, help='path to the GPT2 model')
    
    parser.add_argument('--model_arch_name', type=str, default='gpt2-medium',
                        help='name of the gpt2 model to be used (default: gpt2-medium)')
    
    parser.add_argument('--device', type=str, help='device to train the model (cpu or cuda)')
    eval(parser.parse_args())