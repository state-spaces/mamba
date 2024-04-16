import torch
import transformers
import json

from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm
from torch.utils.data import Dataset


class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.AutoTokenizer, conversation_template: str, max_tokens: int):
        super(ChatDataset, self).__init__()
        data = []
        with open(data_path, "r") as file:
            for line in file:  
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print("json processing exception", e)
                    continue


        data_dict = preprocess(data, tokenizer, conversation_template, max_tokens)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForChatDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

class ChatDataModule():
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, conversation_template, max_tokens: int):

        self.dataset = ChatDataset(tokenizer=tokenizer, data_path=data_path, conversation_template=conversation_template, max_tokens=max_tokens)
        self.data_collator = DataCollatorForChatDataset(tokenizer=tokenizer)
        

def preprocess(conversations: Sequence[Sequence[dict]], tokenizer: transformers.PreTrainedTokenizer, conversation_template: str, max_tokens: int) -> Dict:
    """
    Preprocess the data by tokenizing.
    """
    all_input_ids = []
    all_label_ids = []
    tokenizer.use_default_system_prompt = False

    print("Tokenizing dataset...")
    for conv in tqdm(conversations):
        current_conv = conv["messages"]
        tokenized_responses = []
        for msg in current_conv:
            if msg["role"] == "assistant":
                tokenized_responses.append(tokenizer.encode(msg["content"], add_special_tokens=False))

        tokenized_conv = tokenizer.apply_chat_template(current_conv, chat_template=conversation_template, max_length=max_tokens, truncation=True)
        all_input_ids.append(torch.LongTensor(tokenized_conv))


    return dict(input_ids=all_input_ids, labels=all_input_ids)