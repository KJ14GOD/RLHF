from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    # Wraps one hh-rlhf split. Each item is a preference pair: the same
    # conversation with the response a human preferred ("chosen") and the one
    # they rejected. Tokenization happens in the collate function so the whole
    # batch can be padded together.

    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        return {"chosen": example["chosen"], "rejected": example["rejected"]}


def make_preference_collate(tokenizer, max_length):
    def collate(batch):
        # Tokenizing the batch together pads only to the longest sequence in
        # this batch (dynamic padding), not always to max_length.
        chosen = tokenizer(
            [item["chosen"] for item in batch],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        rejected = tokenizer(
            [item["rejected"] for item in batch],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    return collate
