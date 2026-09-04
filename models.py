# Model classes for the RLHF pipeline: the BERT-based reward model and the
# distilgpt2 policy with a value head.
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from config import CONFIG


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CONFIG.REWARD_MODEL_NAME)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1) # in our case it will be 768 

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # # get the last hidden state which represents the value
        # last_hidden = outputs.hidden_states[-1] # last_hidden.shape is [1, 512, 768]. 1 example in each batch, 512 tokens for each example and 768 numbers for each token
        # last_token_index = attention_mask.sum(dim=1) - 1 # last token in an example
        # batch_index = torch.arange(input_ids.shape[0], device=input_ids.device) # gpu optimization step 
        # final_token_hidden = last_hidden[batch_index, last_token_index] # grabs the hidden vector at the last token position
        # # print(final_token_hidden.shape) # the shape will be [1,768] because the last token has 768 n_embds used to represent it
        # score = self.reward_head(final_token_hidden)


        # BERT is an encoder model. It reads the full sequence bidirectionally.
        # pooler_output is one vector for the whole sequence, based on the [CLS] token.
        pooled_output = outputs.pooler_output # pooled_output.shape is [batch size, 768] for bert-base
        score = self.reward_head(pooled_output)
        return score

# 1. Pass tokens through BERT encoder backbone
# 2. Get pooler_output, which is one vector for the whole sequence
# 3. Pass that pooled vector through a linear layer
# 4. Output one number


class PolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = self.create_tokenizer()
        self.backbone = AutoModelForCausalLM.from_pretrained(CONFIG.POLICY_MODEL_NAME)
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        hidden_size = self.backbone.config.hidden_size  # 768 for distilgpt2

        # The value head predicts for every token position how good the model expects the state to be
        # PPO uses these values to compute advantages
        # maps token's hidden vector (768) to 1
        self.value_head = nn.Linear(hidden_size, 1)

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.POLICY_MODEL_NAME)
        special_tokens = {
            "additional_special_tokens": CONFIG.POLICY_SPECIAL_TOKENS,
            "pad_token": CONFIG.POLICY_PAD_TOKEN,
        }
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    def format_text(self, text):
        return (
            text.replace("Human:", "<|Human|>")
            .replace("\n\nAssistant:", "\n\n<|Assistant|>")
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # logits: what token should come next at each position. shape is [batch, seq_len, vocab_size] vocab_size represents the total number of unique building blocks that the model's tokenizer can read
        logits = outputs.logits # score for every possible token in distilgpt2
        last_hidden = outputs.hidden_states[-1] # last_hidden is the final layer's hidden state for every token. shape is [batch, seq_len, hidden_size]
        values = self.value_head(last_hidden).squeeze(-1) # squeeze the trailing 1. shape is [batch,seq_len]

        return logits, values

