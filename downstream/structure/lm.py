import warnings
warnings.filterwarnings("ignore")
from transformers import EsmTokenizer, EsmModel, BertForMaskedLM, BertModel, AutoConfig
#from mmoe.mmoe_layer import DNATokenizer, MMoeBertForSequenceClassification
#from bert_config import BertConfig
from transformers import Trainer, TrainingArguments, BertTokenizer
import transformers
import sys
sys.path.append("..") 
#from dna_module.tokenization_dna import DNATokenizer
from model.rnalm.rnalm_config import RNALMConfig
from model.rnalm.modeling_rnalm import BertModel as FlashBertModel
#from dnabert2_source.bert_layers import BertModel as DNABERT2

def get_extractor(args):
    '''
    '''

    # the pretrained model names
    name_dict = {'8m': 'esm8m_2parts_5m',
                 '35m': 'esm35m_25parts_31m',
                 '150m': 'esm150m_25parts_31m',
                 '650m': 'esm650m_50parts_100m',
                 '650m-1B': 'esm650m-1B_8clstr_8192',
                 '8m-1B' : 'esm8m_1B',
                 '35m-1B' : 'esm35m_1B',
                 '150m-1B': 'esm150m_1B'
                 }
    if args.model_type == 'rna-fm' or args.model_type == 'esm-rna':
        #assert args.model_scale in name_dict.keys(), print(f'args.model_scale should be in {name_dict.keys()}')

        #extractor = EsmModel.from_pretrained(f'{args.pretrained_lm_dir}/{name_dict[args.model_scale]}/')
        #tokenizer = EsmTokenizer.from_pretrained("/mnt/data/ai4bio/renyuchen/DNABERT/examples/rna_finetune/ssp/vocab_esm_mars.txt")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        if args.train_from_scratch:
            print('Loading esm model')
            print('Train from scratch')
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            extractor = EsmModel(config)
        else:           
            extractor = EsmModel.from_pretrained(args.model_name_or_path)
        
        
        #tokenizer = EsmTokenizer.from_pretrained(f"{args.pretrained_lm_dir}/vocab_esm_mars.txt")
    elif args.model_type == 'esm-protein':
        extractor = EsmModel.from_pretrained(args.model_name_or_path)

        tokenizer = EsmTokenizer.from_pretrained(f"{args.model_name_or_path}/vocab.txt")
    elif args.model_type == 'dnabert':
        
        #print(extractor)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            # trust_remote_code=True,
            )
        if args.train_from_scratch:
            print('Loading dnabert model')
            print('Train from scratch')
            config = MMoeBertConfig.from_pretrained(args.model_name_or_path)
            extractor = BertModel(config)
        else:
            extractor = BertModel.from_pretrained(
            args.model_name_or_path,
            )
    elif args.model_type == 'dnabert2':
        
        #print(extractor)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            # trust_remote_code=True,
            )
        if args.train_from_scratch:
            print('Loading dnabert model')
            print('Train from scratch')
            config = MMoeBertConfig.from_pretrained(args.model_name_or_path)
            extractor = DNABERT2(config)
        else:
            extractor = DNABERT2.from_pretrained(
            args.model_name_or_path,
            )    
    elif args.model_type == 'rnalm':
        print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            # trust_remote_code=True,
            )
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(tokenizer)
        if args.train_from_scratch:
            print('Loading rnabert model')
            print('Train from scratch')
            config = RNALMConfig.from_pretrained(args.model_name_or_path)
            extractor = FlashBertModel(config)
        else:           
            extractor = FlashBertModel.from_pretrained(
                args.model_name_or_path,
            )
        
    elif args.model_type == 'mmoe':
        pass
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print(extractor)
    print(tokenizer)
    return extractor, tokenizer


def unitest(args):

    extractor, tokenizer = get_extractor(args)

    # replace 'U' with 'T'
    seqs = ['ATGCATGCATGCATGCATGC']

    max_len = 128

    data_dict = tokenizer.batch_encode_plus(seqs,
                                            padding='max_length',
                                            max_length=max_len,
                                            truncation=True,
                                            return_tensors='pt')

    input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']

    output = extractor(input_ids=input_ids, attention_mask=attention_mask)

    #print(output.keys())

    #print(output['last_hidden_state'].shape)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_lm_dir", type=str, default='/public/home/taoshen/data/rna/mars_fm_data/mars_esm_preckpts')

    parser.add_argument("--model_scale", type=str, default='8m')

    args = parser.parse_args()

    unitest(args)