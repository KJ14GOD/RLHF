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


import argparse
import copy
import os

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
PPO_MODEL_PATH = "ppo_policy_model.pt"
POLICY_SPECIAL_TOKENS = ["<|Human|>", "<|Assistant|>"]
POLICY_PAD_TOKEN = "<|pad|>"
PPO_LEARNING_RATE = 1e-6
PPO_CLIP_RANGE = 0.2
VALUE_CLIP_RANGE = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
KL_COEF = 0.05
GAMMA = 1.0
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 1.0
MAX_NEW_TOKENS = 64
PPO_EPOCHS = 4

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_reward_model():
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
    best_test_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        reward_model.train()
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

        avg_test_loss = test_loss / test_examples
        test_accuracy = test_correct / test_examples
        print("Test average loss:", avg_test_loss)
        print("Test accuracy:", test_accuracy)

        # Save based on the held-out preference loss; accuracy is only a diagnostic.
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
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
        self.tokenizer = self.create_tokenizer()
        self.backbone = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME)
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        hidden_size = self.backbone.config.hidden_size  # 768 for distilgpt2

        # The value head predicts for every token position how good the model expects the state to be
        # PPO uses these values to compute advantages
        # maps token's hidden vector (768) to 1
        self.value_head = nn.Linear(hidden_size, 1)

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
        special_tokens = {
            "additional_special_tokens": POLICY_SPECIAL_TOKENS,
            "pad_token": POLICY_PAD_TOKEN,
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


def get_answer_stats(logits, values, input_ids, prompt_length):
    # Token j is predicted by the logits/value at position j - 1.
    answer_ids = input_ids[:, prompt_length:]
    prediction_logits = logits[:, prompt_length - 1:-1, :]
    answer_values = values[:, prompt_length - 1:-1]
    answer_log_probs = F.log_softmax(prediction_logits, dim=-1).gather(
        dim=-1,
        index=answer_ids.unsqueeze(-1),
    ).squeeze(-1)
    entropy = torch.distributions.Categorical(logits=prediction_logits).entropy()
    return answer_log_probs, answer_values, entropy


def generate_rollout(policy, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0):
    tokenizer = policy.tokenizer
    device = next(policy.parameters()).device

    formatted_prompt = policy.format_text(prompt)
    tokens = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    prompt_length = input_ids.shape[1]
    policy.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = policy(input_ids, attention_mask)
            final_logits = logits[:, -1, :] / temperature
            next_token_id = torch.distributions.Categorical(logits=final_logits).sample()
            next_token_id = next_token_id.unsqueeze(-1)

            # Add the selected token to the sequence.
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # The generated token is real text, so its mask value is 1.
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)] , dim=1,)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    generated_ids = input_ids[:, prompt_length:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    with torch.no_grad():
        logits, values = policy(input_ids, attention_mask)
        old_log_probs, old_values, _ = get_answer_stats(
            logits,
            values,
            input_ids,
            prompt_length,
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_length": prompt_length,
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "old_log_probs": old_log_probs,
        "old_values": old_values,
    }


def create_reference_model(policy):
    reference_model = copy.deepcopy(policy.backbone)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    return reference_model


def get_reference_log_probs(reference_model, rollout):
    with torch.no_grad():
        outputs = reference_model(
            input_ids=rollout["input_ids"],
            attention_mask=rollout["attention_mask"],
        )
        answer_ids = rollout["generated_ids"]
        prediction_logits = outputs.logits[:, rollout["prompt_length"] - 1:-1, :]
        return F.log_softmax(prediction_logits, dim=-1).gather(
            dim=-1,
            index=answer_ids.unsqueeze(-1),
        ).squeeze(-1)


def load_reward_model(device):
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} was not found. Run `uv run python train.py reward` first."
        )

    reward_model = RewardModel().to(device)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    reward_model.load_state_dict(state_dict)
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad = False
    return reward_model


def score_response(reward_model, reward_tokenizer, prompt, generated_text):
    device = next(reward_model.parameters()).device
    text = prompt + generated_text
    tokens = reward_tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        score = reward_model(
            tokens["input_ids"].to(device),
            tokens["attention_mask"].to(device),
        )
    return score.squeeze()


def compute_advantages(old_values, token_rewards):
    advantages = torch.zeros_like(token_rewards)
    last_advantage = torch.zeros(
        token_rewards.shape[0],
        device=token_rewards.device,
    )

    for token_index in reversed(range(token_rewards.shape[1])):
        if token_index == token_rewards.shape[1] - 1:
            next_value = torch.zeros_like(last_advantage)
        else:
            next_value = old_values[:, token_index + 1]

        delta = (
            token_rewards[:, token_index]
            + GAMMA * next_value
            - old_values[:, token_index]
        )
        last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[:, token_index] = last_advantage

    returns = advantages + old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    return advantages, returns


def ppo_update(policy, optimizer, rollout, advantages, returns):
    old_log_probs = rollout["old_log_probs"]
    old_values = rollout["old_values"]
    metrics = {}
    policy.train()

    for _ in range(PPO_EPOCHS):
        logits, values = policy(rollout["input_ids"], rollout["attention_mask"])
        log_probs, current_values, entropy = get_answer_stats(
            logits,
            values,
            rollout["input_ids"],
            rollout["prompt_length"],
        )

        ratio = torch.exp(log_probs - old_log_probs)
        unclipped_objective = ratio * advantages
        clipped_objective = torch.clamp(
            ratio,
            1.0 - PPO_CLIP_RANGE,
            1.0 + PPO_CLIP_RANGE,
        ) * advantages
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()

        clipped_values = old_values + torch.clamp(
            current_values - old_values,
            -VALUE_CLIP_RANGE,
            VALUE_CLIP_RANGE,
        )
        value_loss = 0.5 * torch.max(
            (current_values - returns).pow(2),
            (clipped_values - returns).pow(2),
        ).mean()
        entropy_bonus = entropy.mean()
        loss = (
            policy_loss
            + VALUE_LOSS_COEF * value_loss
            - ENTROPY_COEF * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_bonus.item(),
                "clip_fraction": ((ratio - 1.0).abs() > PPO_CLIP_RANGE).float().mean().item(),
            }

    return metrics


def train_ppo(prompts, num_steps=10):
    device = get_device()
    print("Using device:", device)
    reward_model = load_reward_model(device)
    policy = PolicyModel().to(device)
    reference_model = create_reference_model(policy)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=PPO_LEARNING_RATE)

    for step in range(num_steps):
        prompt = prompts[step % len(prompts)]
        rollout = generate_rollout(policy, prompt)
        reference_log_probs = get_reference_log_probs(reference_model, rollout)
        reward_score = score_response(
            reward_model,
            reward_tokenizer,
            prompt,
            rollout["generated_text"],
        )

        with torch.no_grad():
            log_ratio = rollout["old_log_probs"] - reference_log_probs
            token_rewards = -KL_COEF * log_ratio
            token_rewards[:, -1] += reward_score
            advantages, returns = compute_advantages(
                rollout["old_values"],
                token_rewards,
            )

        metrics = ppo_update(policy, optimizer, rollout, advantages, returns)
        print(
            f"step={step + 1}/{num_steps} "
            f"reward={reward_score.item():.4f} "
            f"kl={log_ratio.mean().item():.4f} "
            f"loss={metrics['loss']:.4f} "
            f"generated={rollout['generated_text']!r}"
        )

    torch.save(policy.state_dict(), PPO_MODEL_PATH)
    print("Saved PPO policy:", PPO_MODEL_PATH)
    return policy


def test_generate_rollout():
    device = get_device()
    policy = PolicyModel().to(device)

    rollout = generate_rollout(
        policy,
        "Human: What is reinforcement learning?\n\nAssistant:",
    )

    print("Prompt length:", rollout["prompt_length"])
    print("Full sequence shape:", rollout["input_ids"].shape)
    print("Generated token IDs:", rollout["generated_ids"])
    print("Generated text:", rollout["generated_text"])
    print("Answer log probs shape:", rollout["old_log_probs"].shape)
    print("Answer values shape:", rollout["old_values"].shape)



def test_policy_model():
    device = get_device()
    print("Using device:", device)

    policy = PolicyModel().to(device)
    tokenizer = policy.tokenizer
    policy.eval()

    prompt = policy.format_text("Human: What is reinforcement learning?\n\nAssistant:")
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        logits, values = policy(input_ids, attention_mask)

    print("input_ids shape:",(input_ids.shape))  # [1, seq_len]
    print("logits shape:   ",(logits.shape))      # [1, seq_len, vocab_size]
    print("values shape:   ",(values.shape))      # [1, seq_len]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the RLHF PPO pipeline.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("rollout", "policy", "reward", "ppo"),
        default="rollout",
    )
    parser.add_argument("--ppo-steps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "reward":
        train_reward_model()
    elif args.mode == "ppo":
        train_ppo(
            prompts=[
                "Human: What is reinforcement learning?\n\nAssistant:",
                "Human: Explain gradient descent simply.\n\nAssistant:",
                "Human: What makes an answer helpful?\n\nAssistant:",
            ],
            num_steps=args.ppo_steps,
        )
    elif args.mode == "policy":
        test_policy_model()
    else:
        test_generate_rollout()
