import os

import pandas as pd
import pytorch_lightning as pl
from hanziconv import HanziConv

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
        train_path = os.path.join(path, 'train')
        test_path = os.path.join(path, 'test')

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

pl.seed_everything(42)

if __name__ == '__main__':
    model = T5BaseModel()
    model_path = os.path.join(os.getcwd(), 'mengzi-t5-base')
    model.load_model(model_path, use_gpu=True)
    dataset = CoupletDataset(path=COUPLET_PATH, prompt=COUPLET_PROMPT)

    model.train(
        train_df = dataset.train_df,
        eval_df=dataset.test_df,
        source_max_token_len=MAX_IN_TOKENS,
        target_max_token_len=MAX_OUT_TOKENS,
        batch_size=64,
        max_epochs=5,
        use_gpu=True,
        save_dir="./t5-couplet"
    )
