import torch
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForMaskedLM
 
from utils import get_verbalization_ids


class LabelEncoder(object):
    def __init__(self, tokenizer, pvp, label_list):
        # Record prompt tokens
        pattern_token_set, pattern_token_indices = set(), []
        # RoBERTa tokenizer is initiated from GPT2Tokenizer,
        # and it tokenizes same words differently in different positions:
        # e.g.  'Hello world!' -> ['Hello', 'Ġworld', '!'];
        #       'Hello', 'world' -> ['Hello'], ['world']
        # So we need to add prefix space to simulate true situations
        kwargs = {'add_prefix_space': True} if isinstance(
            tokenizer, GPT2Tokenizer) else {}
        # Record label tokens
        label_token_ids = []
        for label_idx, label in enumerate(label_list): #["entailment", "not_entailment"]
            verbalizers = pvp.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(
                    verbalizer, tokenizer, force_single_token=True)
                assert verbalizer_id != tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                label_token_ids.append(verbalizer_id)

        assert len(pattern_token_set) < 50 and len(label_token_ids) < 49
        #print("prompt encoder")
        #import pdb; pdb.set_trace()
        # Convert tokens in manual prompt / label to unused tokens
        # Note that `AlbertTokenizer` or `RobertaTokenizer` doesn't have a `vocab` attribute
        if hasattr(tokenizer, 'vocab') and '[unused0]' in tokenizer.vocab:
            # BERT
            self.label_convert = {token_id: tokenizer.vocab['[unused%s]' % idx]
                                    for idx, token_id in enumerate(pattern_token_set)}

        else:
            # ALBERT, RoBERTa
            start_idx = tokenizer.vocab_size - 100
            self.label_convert = {token_id: start_idx + idx
                                    for idx, token_id in enumerate(pattern_token_set)}

        # Convert mlm logits to cls logits
        self.vocab_size = tokenizer.vocab_size
        self.m2c_tensor = torch.tensor( 
            list(self.label_convert.values()), dtype=torch.long) #包含label unused token的tensor


    def init_embed(self, model, random_=False):
        w = model.get_input_embeddings().weight.data
        for origin_id, convert_id in self.label_convert.items():
            if random_:
                max_val = w[convert_id].abs().max()
                w[convert_id].uniform_(-max_val, max_val)
            else:
                w[convert_id] = w[origin_id] #可以改动


    def add_embed_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = list(self.pattern_convert.values()) + \
            list(self.label_convert.values())
        grad_mask = torch.zeros((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 1.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)
    
    def add_reverse_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = list(self.pattern_convert.values()) + \
            list(self.label_convert.values())
        grad_mask = torch.ones((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 0.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def get_replace_embeds(self, word_embeddings):
        return word_embeddings(self.lookup_tensor.to(word_embeddings.weight.device))

    def convert_mlm_logits_to_cls_logits(self, mlm_labels, logits):
        return torch.index_select(logits[mlm_labels != -1], -1, self.m2c_tensor.to(logits.device))

class TransformerModelWrapper(object):
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config

        # tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_fast=False)

        self.pvp = PVPS[config.task_name](self, config.pattern_id)
        self.model = AutoModelForMaskedLM.from_pretrained()
        self.task_helper = load_task_helper(config.task_name, self)
        self.label_map = {label: i for i,
                          label in enumerate(self.config.label_list)}

        self.encoder = LabelEncoder(
            self.tokenizer, self.pvp, config.label_list)
        # Random init prompt tokens HERE!
        self.encoder.init_embed(self.model.model, random_=False)

        if config.device == 'cuda':
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
            # Use automatic mixed precision for faster training
            # self.scaler = GradScaler()