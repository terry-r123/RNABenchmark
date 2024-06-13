import warnings
warnings.filterwarnings("ignore")
from transformers import EsmTokenizer, EsmModel, BertForMaskedLM, BertModel, AutoConfig, BertTokenizer

from transformers import Trainer, TrainingArguments, BertTokenizer
import transformers
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = (os.path.dirname(os.path.dirname(current_path)))
print(parent_dir)
sys.path.append(parent_dir)

from model.rnalm.rnalm_config import RnaLmConfig
from model.rnalm.modeling_rnalm import RnaLmModel 
from model.rnalm.rnalm_tokenizer import RnaLmTokenizer
from model.rnafm.modeling_rnafm import RnaFmModel
from model.rnabert.modeling_rnabert import RnaBertModel
from model.rnamsm.modeling_rnamsm import RnaMsmModel
from model.splicebert.modeling_splicebert import SpliceBertModel
from model.utrbert.modeling_utrbert import UtrBertModel
from model.utrlm.modeling_utrlm import UtrLmModel
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
def get_extractor(args):
    '''
    '''
    if args.model_type == 'rnalm':
        if args.token_type != 'single':
            tokenizer = EsmTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=args.token_type
                )
        else:
            tokenizer = RnaLmTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=args.token_type
                )
        print(tokenizer)
        if args.train_from_scratch:
            print(f'Train from scratch {args.model_type} model')
            config = RnaLmConfig.from_pretrained(args.model_name_or_path,
            attn_implementation=args.attn_implementation,)
            extractor = RnaLmModel(config)
        else:           
            extractor = RnaLmModel.from_pretrained(
                args.model_name_or_path,
                token_type=args.token_type,
                attn_implementation=args.attn_implementation,
            )
    elif args.model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        tokenizer = OpenRnaLMTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        if args.model_type == 'rna-fm':      
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = RnaFmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
 
            )     
        elif args.model_type == 'rnabert':      
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = RnaBertModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,

            )     
        elif args.model_type == 'rnamsm':
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = RnaMsmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )        
        elif 'splicebert' in args.model_type:
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = SpliceBertModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,

            )       
        elif 'utrbert' in args.model_type:
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = UtrBertModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )  
        elif 'utr-lm' in args.model_type:
            print(args.model_type)
            print(f'Loading {args.model_type} model')
            extractor = UtrLmModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )  
        
    print(tokenizer)
    return extractor, tokenizer

