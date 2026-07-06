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


from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

#modify token length based on gpu
REWARD_MODEL_NAME = "google-bert/bert-base-cased"
POLICY_MODEL_NAME = "distilgpt2"
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
TRAIN_EXAMPLES = 500
TEST_EXAMPLES = 500
PRINT_EVERY = 10
NUM_EPOCHS = 2
BEST_MODEL_PATH = "best_reward_model.pt"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    device = get_device()
    print("Using device:", device)

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    tokenizer.truncation_side = "left"
    
    # For batching, we reuse the end-of-text token as padding.

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(dataset)
    print("Available splits:", dataset.keys())
    print("Tokenizer truncation side:", tokenizer.truncation_side)

    train_data = dataset["train"]
    print("One row has these columns:", train_data.column_names)
    print("Number of training examples:", len(train_data))

    select_sample = train_data.select(range(TRAIN_EXAMPLES))
    print("select sample", select_sample)

    print("************************************************")

    test_data = dataset["test"]
    print("One row has these columns:", test_data.column_names)
    print("Number of training examples:", len(test_data))

    test_sample = test_data.select(range(TEST_EXAMPLES))
    print("test sample", test_sample)

    reward_model = RewardModel().to(device)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=LEARNING_RATE)
    reward_model.train()
    best_test_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        print(f"\nStarting epoch {epoch + 1}/{NUM_EPOCHS}")

        for i in range(len(select_sample)):
            example = select_sample[i]
            chosen_tokens = tokenizer(
                example["chosen"],
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            rejected_tokens = tokenizer(
                example["rejected"],
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # attention mask marks 1 as a real token and 0 as a padding token. Tells the model this is text and this is padding
            # .shape represents [batch size, sequence length] 

            chosen_input_ids = chosen_tokens["input_ids"].to(device)
            chosen_attention_mask = chosen_tokens["attention_mask"].to(device)
            rejected_input_ids = rejected_tokens["input_ids"].to(device)
            rejected_attention_mask = rejected_tokens["attention_mask"].to(device)

            chosen_score = reward_model(
                chosen_input_ids,
                chosen_attention_mask,
            )
            rejected_score = reward_model(
                rejected_input_ids,
                rejected_attention_mask,
            )

            loss = -F.logsigmoid(chosen_score - rejected_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct = (chosen_score > rejected_score).float().item()
                total_correct += correct
                total_examples += 1
                total_loss += loss.item()

            if (i + 1) % PRINT_EVERY == 0:
                avg_loss = total_loss / total_examples
                accuracy = total_correct / total_examples
                print(
                    f"epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"step {i + 1}/{len(select_sample)} "
                    f"loss={avg_loss:.4f} accuracy={accuracy:.4f} "
                    f"chosen_score={chosen_score.item():.4f} "
                    f"rejected_score={rejected_score.item():.4f}"
                )

        print("Epoch average loss:", total_loss / total_examples)
        print("Epoch training accuracy:", total_correct / total_examples)

        reward_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_examples = 0

        # evaluation does not train the model. It only checks whether chosen_score > rejected_score
        # on examples the reward model did not see during training.

        with torch.no_grad():
            for i in range(len(test_sample)):
                example = test_sample[i]
                chosen_tokens = tokenizer(
                    example["chosen"],
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                rejected_tokens = tokenizer(
                    example["rejected"],
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                # attention mask marks 1 as a real token and 0 as a padding token. Tells the model this is text and this is padding
                # .shape represents [batch size, sequence length]

                chosen_input_ids = chosen_tokens["input_ids"].to(device)
                chosen_attention_mask = chosen_tokens["attention_mask"].to(device)
                rejected_input_ids = rejected_tokens["input_ids"].to(device)
                rejected_attention_mask = rejected_tokens["attention_mask"].to(device)

                chosen_score = reward_model(
                    chosen_input_ids,
                    chosen_attention_mask,
                )
                rejected_score = reward_model(
                    rejected_input_ids,
                    rejected_attention_mask,
                )

                loss = -F.logsigmoid(chosen_score - rejected_score).mean()
                correct = (chosen_score > rejected_score).float().item()

                test_correct += correct
                test_examples += 1
                test_loss += loss.item()

        test_accuracy = test_correct / test_examples
        print("Test average loss:", test_loss / test_examples)
        print("Test accuracy:", test_accuracy)

        # Save the best reward model weights so PPO can later load the best scorer.
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(reward_model.state_dict(), BEST_MODEL_PATH)
            print("Saved new best reward model:", BEST_MODEL_PATH)

        reward_model.eval()



class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(REWARD_MODEL_NAME)
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
        self.backbone = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME)
        hidden_size = self.backbone.config.hidden_size  # 768 for distilgpt2

        # The value head predicts for every token position how good the model expects the state to be
        # PPO uses these values to compute advantages
        # maps token's hidden vector (768) to 1
        self.value_head = nn.Linear(hidden_size, 1)

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


def test_policy_model():
    device = get_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = PolicyModel().to(device)
    policy.eval()

    prompt = "Human: What is reinforcement learning?\n\nAssistant:"
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        logits, values = policy(input_ids, attention_mask)

    print("input_ids shape:",(input_ids.shape))  # [1, seq_len]
    print("logits shape:   ",(logits.shape))      # [1, seq_len, vocab_size]
    print("values shape:   ",(values.shape))      # [1, seq_len]


if __name__ == "__main__":
    test_policy_model()