# detailed pipeline
# 1. policy model - llm i am training with ppo
# 2 reference model - frozen copy of llm for kl divergence
# 3 reward model - a model trained to score responses as better or worse using Anthropic dataset
# 4 value head - a small head on top of the policy model that predicts expected reward for PPO

#rlhf has 3 main components
# 1. train reward model
# 2. generate responses from policy model
# 3. use ppo to improve policy using reward model scores

# policy model generates a response
# reward model scores the response
# ppo updates the policy model to make high scoring responses more likely
# reference model is used to calculate kl divergence so that it doesn't go too far from the reference model 

# train reward model
# create policy model
# generate rollouts
# score response
# ppo update


from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn

#modify token length based on gpu
MODEL_NAME = "distilgpt2"
MAX_LENGTH = 512

def main():
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.truncation_side = "left"

    # GPT-2 style tokenizers do not have a padding token by default.
    # For batching, we reuse the end-of-text token as padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(dataset)
    print("Available splits:", dataset.keys())
    print("Tokenizer truncation side:", tokenizer.truncation_side)

    train_data = dataset["train"]
    print("One row has these columns:", train_data.column_names)
    print("Number of training examples:", len(train_data))

    first_example = train_data[0]

    # reward model reads tensors
    chosen_tokens = tokenizer(
        first_example["chosen"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    rejected_tokens = tokenizer(
        first_example["rejected"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # attention mask marks 1 as a real token and 0 as a padding token. tells the model this is text and this is padding
    # .shape represents [batch size, sequence length]

    reward_model = RewardModel()
    reward_model.eval()
    
    with torch.no_grad():
        chosen_score = reward_model(
            chosen_tokens["input_ids"],
            chosen_tokens["attention_mask"],
        )
        rejected_score = reward_model(
            rejected_tokens["input_ids"],
            rejected_tokens["attention_mask"],
        )
    
    
    print("chosen score:", chosen_score)
    print("rejected score:", rejected_score)

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1) # in our case it will be 768 

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # get the last hidden state which represents the value
        last_hidden = outputs.hidden_states[-1]
        last_token_index = attention_mask.sum(dim=1) - 1 # last token in an example
        batch_index = torch.arange(input_ids.shape[0]) # gpu optimization step 
        final_token_hidden = last_hidden[batch_index, last_token_index] # grabs the hidden vector at the last token position

        score = self.reward_head(final_token_hidden)
        return score

# 1. Pass tokens through distilgpt2 backbone
# 2. Get hidden states for every token
# 3. Pick the final real token using attention_mask
# 4. Pass that hidden vector through a linear layer
# 5. Output one number


if __name__ == "__main__":
    main()
