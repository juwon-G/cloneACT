from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast


def load_model_tokenizer(model_name, source_lang, target_lang, device):
    # load NMT model
    if model_name=="marian":
        name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    elif model_name=="mbart":
        name = 'facebook/mbart-large-50-one-to-many-mmt'
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
    
    # load tokenizer
    if "marian" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    elif model_name=="mbart":
        tokenizer = MBart50TokenizerFast.from_pretrained(name)
        dict_lang = {"en":"en_XX", "de":"de_DE", "fr":"fr_XX", "zh":"zh_CN", "cs":"cs_CZ", "ru":"ru_RU"}
        tokenizer.src_lang = dict_lang[source_lang]
        tokenizer.tgt_lang = dict_lang[target_lang]
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang ]
        
    tokenizer.model_max_length = 512


    return model, tokenizer

