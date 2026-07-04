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
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    


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


    print("\n" + "=" * 80)
    print("Tokenized first example")
    print("chosen input_ids shape:", chosen_tokens["input_ids"].shape)
    print("chosen attention_mask shape:", chosen_tokens["attention_mask"].shape)
    print("rejected input_ids shape:", rejected_tokens["input_ids"].shape)
    print("rejected attention_mask shape:", rejected_tokens["attention_mask"].shape)
    print("first 20 chosen token ids:", chosen_tokens["input_ids"][0, :20])
    print("decoded first 20 chosen tokens:")
    print(tokenizer.decode(chosen_tokens["input_ids"][0, :20]))


    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids = chosen_tokens["input_ids"],
            attention_mask = chosen_tokens["attention_mask"],
            output_hidden_states=True,
        )
    
    # get the last hidden state which represents the value
    last_hidden = outputs.hidden_states[-1] 
    last_token_index = chosen_tokens["attention_mask"].sum(dim=1) - 1 # last token in an example 

    # gpu optimization step
    batch_index = torch.arange(chosen_tokens["input_ids"].shape[0])
    final_token_hidden = last_hidden[batch_index, last_token_index] # grabs the hidden vector at the last token position
    
    reward_head = nn.Linear(final_token_hidden.shape[-1], 1)
    score = reward_head(final_token_hidden)

    print("last hidden shape:", last_hidden.shape)
    print("last token index:", last_token_index)
    print("final token hidden shape:", final_token_hidden.shape)
    print("reward score shape:", score.shape)
    print("reward score:", score)


# 1. Pass tokens through distilgpt2 backbone
# 2. Get hidden states for every token
# 3. Pick the final real token using attention_mask
# 4. Pass that hidden vector through a linear layer
# 5. Output one number


if __name__ == "__main__":
    main()
