import os

import pandas as pd
import torch
from hanziconv import HanziConv
from rich.progress import track

from config import COUPLET_PATH
from model.t5_base import T5BaseModel

MAX_SEQ_LEN = 32
COUPLET_PROMPT = "对联: "
MAX_IN_TOKENS = MAX_SEQ_LEN + len(COUPLET_PROMPT)
MAX_OUT_TOKENS = MAX_SEQ_LEN


class CoupletDataset():
    def __init__(
        self,
        path=COUPLET_PATH,
        prompt="对联: "
    ):
        self.prompt = prompt
        train_path = os.path.join(COUPLET_PATH, 'train')
        test_path = os.path.join(COUPLET_PATH, 'test')

        self.train_df = self.build_dataset(train_path)
        self.test_df = self.build_dataset(test_path)

    def build_dataset(self, path):
        ins = []
        outs = []

        for i in ['in', 'out']:
            fpath = os.path.join(path, f"{i}.txt")
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    clean_line = line.strip().replace(' ', '').replace('\n', '').replace('\r', '')
                    clean_line = clean_line[:MAX_SEQ_LEN]
                    clean_line = HanziConv.toSimplified(clean_line)
                    if i == 'in':
                        ins.append(clean_line)
                    elif i == 'out':
                        outs.append(clean_line)
        data_dict = {
            'source_text': ins,
            'target_text': outs
        }
        data_df = pd.DataFrame(data_dict)
        data_df['source_text'] = self.prompt + data_df['source_text']

        invalid_lines = []
        for idx in range(len(data_df)):
            if len(data_df['source_text'].values[idx]) != len(data_df['target_text'].values[idx]) + len(self.prompt):
                print("mismatch:", data_df['source_text'].values[idx], data_df['target_text'].values[idx])
                invalid_lines.append(idx)
        data_df = data_df.drop(index=invalid_lines)

        return data_df


class MengZiCouplet(T5BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda')
        self.from_pretrained(model_name='Langboat/mengzi-t5-base')


if __name__ == '__main__':
    model = MengZiCouplet()
    model.tokenizer("对联: 你好")
    # dataset = CoupletDataset(path=COUPLET_PATH, prompt=COUPLET_PROMPT)
    # print(dataset.train_df.head())
