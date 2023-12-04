import datetime
import json
import os
import sys

from datasets import load_dataset, Features, Value, Dataset
import glob
import random
from typing import List
import matplotlib.pyplot as plt

from tqdm import trange
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, GPTJForCausalLM, AutoConfig, AutoModelForCausalLM
from torch.optim import AdamW

BATCH_SIZE = 1
IGNORED_PAD_IDX = -100
TOKEN_BOS = '<|endoftext|>'
TOKEN_EOS = '<|END|>'
TOKEN_PAD = '<|pad|>'
TOKEN_SEP = '\n\n####\n\n'
EPOCHS = 30
plm = "EleutherAI/pythia-70m-deduped"


class BatchSampler:
    def __init__(self, data, batch_size):
        self.pooled_indices = []
        self.data = data
        self.batch_size = batch_size
        self.len = len(list(data))

    def __iter__(self):
        self.pooled_indices = []
        # indices = [(index, len(data["content"])) for index, data in enumerate(self.data)]
        indices = []
        for index, data in enumerate(self.data):
            if (data["content"] is not None):
                indices.append((index, len(data["content"])))
            # else:
            #     indices.append((index, 0))

        random.shuffle(indices)
        for i in range(0, len(indices), BATCH_SIZE * 100):
            self.pooled_indices.extend(sorted(indices[i:i + BATCH_SIZE * 100], key=lambda x: x[1], reverse=True))
        self.pooled_indices = [x[0] for x in self.pooled_indices]

        for i in range(0, len(self.pooled_indices), BATCH_SIZE):
            yield self.pooled_indices[i:i + BATCH_SIZE]

    def __len__(self):
        return self.len + self.batch_size - 1


def createTokenizer():
    # special_tokens_dict: List[str] = [bos, eos, pad, sep]
    special_tokens_dict = {'eos_token': TOKEN_EOS, 'bos_token': TOKEN_BOS, 'pad_token': TOKEN_PAD,
                           'sep_token': TOKEN_SEP}

    tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
    tokenizer.padding_side = 'left'

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    return tokenizer


def createModel(tokenizer):
    config = AutoConfig.from_pretrained(plm,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        sep_token_id=tokenizer.sep_token_id,
                                        output_hidden_states=False,)
    model = AutoModelForCausalLM.from_pretrained(plm, revision="step3000", config=config)
    model.resize_token_embeddings(len(tokenizer))
    return model


def collate_batch(batch):
    texts = []
    for single_batch in batch:
        batch_input = single_batch['content']
        batch_answer = single_batch['json_answer']
        answers = json.loads(batch_answer)
        texts.append(f'{TOKEN_BOS}{batch_input}{TOKEN_SEP}{json.dumps(answers)}{TOKEN_EOS}')

    encoded_seq = tokenizer(texts, padding=True)
    indexed_tks = torch.tensor(encoded_seq['input_ids'])
    attention_mask = torch.tensor(encoded_seq['attention_mask'])
    encoded_label = torch.tensor(encoded_seq['input_ids'])
    encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX
    return indexed_tks, encoded_label, attention_mask


if __name__ == '__main__':
    args = sys.argv
    dataset_file_path = args[1]
    model_save_path = args[2]
    max_split_size_mb = int(args[3])
    validation_file_path = args[4]

    if len(args) == 6:
        model_file_path = args[5]
    else:
        model_file_path = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_file_paths = glob.glob(dataset_file_path)
    print("訓練集:", dataset_file_paths)
    print("模型儲存路徑:", model_save_path)
    print("已訓練模型路徑:", model_file_path)
    print("最大切割顯存:", max_split_size_mb)
    print("torch.device 是", device)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size_mb}"
    os.system("echo %PYTORCH_CUDA_ALLOC_CONF%")

    tokenizer = createTokenizer()

    model = createModel(tokenizer)
    if model_file_path:
        model.load_state_dict(torch.load(model_file_path))

    model.to(device)

    print("模型儲存測試:")
    try:
        torch.save(model.state_dict(), model_save_path + ".test")
    except Exception as e:
        print(e)
        exit(1)

    delimiter = bytes([0x04, 0x04]).decode('utf-8')

    data_files = {"train": dataset_file_paths, "test": validation_file_path}

    dataset = load_dataset("csv",
                           data_files=data_files,
                           delimiter=delimiter,
                           features=Features({
                               'json_answer': Value('string'),
                               'content': Value('string')
                           }), column_names=['json_answer', 'content'])

    train_data = list(dataset['train'])

    bucket_train_dataloader = DataLoader(train_data, batch_sampler=BatchSampler(train_data, BATCH_SIZE),
                                         collate_fn=collate_batch, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for _ in trange(EPOCHS, desc="Epoch"):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []

        for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
            #  要進模型的資料丟進GPU
            seqs = seqs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            model.zero_grad()

            outputs = model(seqs, labels=labels, attention_mask=masks)  # 模型向前傳播
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()
            total_loss += loss.item()
            loss.backward()  # 模型向後傳播計算Loss
            optimizer.step()  # 更新梯度
        avg_train_loss = total_loss / len(bucket_train_dataloader)
        print("Average train loss:{}".format(avg_train_loss))

    # torch.cuda.memory_summary(device=None, abbreviated=False)
    torch.save(model.state_dict(), model_save_path)
