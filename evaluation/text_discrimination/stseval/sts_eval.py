from __future__ import absolute_import, division, unicode_literals
import sys
import logging
import torch

# Set SENTEVAL PATH
PATH_SENTEVAL = './SentEval'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

def sts_eval(args, transfer_tasks, model, tokenizer):
    # Set up logger and device
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    def prepare(params, samples):
        return 


    def batcher(params, batch):
        # inputs_ids= []
        # for text in batch:
        #     tokens = params.tokenizer.tokenize(text, max_length=max_seqlen, truncation=True)
        #     tokens_ids = params.tokenizer.convert_tokens_to_ids(tokens[:max_seqlen])
        #     inputs_ids.append(tokens_ids)
        sentences = [' '.join(s) for s in batch]
        features = params.tokenizer.batch_encode_plus(
            sentences,
            max_length=params['max_length'],
            return_tensors='pt',
            #         padding=True,
            padding='max_length',
            truncation=True
        )

        device = params['device']
        input_ids, attention_mask = features['input_ids'].to(device), features['attention_mask'].to(device)

        if params['model_name'] == "BART":
            model_output = params.transformer.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_states = model_output['encoder_last_hidden_state']
        elif params['model_name'] == "T5":
            model_output = params.transformer.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, decoder_input_ids=input_ids)
            last_hidden_states = model_output['encoder_last_hidden_state']
        else:
            gpt_output = params.transformer.forward(input_ids=input_ids, output_hidden_states=True)
            last_hidden_states = gpt_output.hidden_states[-1]

        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(last_hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        return embeddings.detach().cpu().numpy()

    # define senteval params
    params_senteval = {'task_path': args.path_to_sts_data, 'usepytorch': True, 'kfold': 10}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
    params_senteval['device'] = args.device
    params_senteval['model_name'] = args.gpt
    params_senteval['deepspeed'] = args.eval_deepspeed_ckpt
    params_senteval['transformer'] = model
    params_senteval['tokenizer'] = tokenizer
    params_senteval['max_length'] = args.max_length
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    eval_results = se.eval(transfer_tasks)

    return eval_results





    
    
    
    

