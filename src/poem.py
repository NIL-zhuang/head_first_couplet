import os

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from model.t5_base import T5BaseModel

TEST_FLOW = False
AUTHOR_PROMPT = "模仿："
TITLE_PROMPT = "作诗："
EOS_TOKEN = "</s>"

IMITATE_AUTHOR = True
IMITATE_TITLE = True

MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 10
MAX_CONTENT_CHAR = 64

MAX_IN_TOKENS = len(TITLE_PROMPT) + MAX_TITLE_CHAR + 1 + len(AUTHOR_PROMPT) + MAX_AUTHOR_CHAR + MAX_CONTENT_CHAR
MAX_OUT_TOKENS = MAX_CONTENT_CHAR

if TEST_FLOW:
    POEM_PATH = os.path.join(os.getcwd(), r'data/poems/poems_test.csv')
else:
    POEM_PATH = os.path.join(os.getcwd(), r'data/poems/poem.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'mengzi-t5-base')


class PoemDataset():
    def __init__(
        self,
        path=POEM_PATH,
    ) -> None:
        self.data = pd.read_csv(path)

    def build_dataset(self, imitate_author=True, imitate_title=True):
        if not imitate_author and not imitate_title:
            raise ValueError("You must set at least one of imitate_author or imitate_title to True")
        dfc = self.data.copy()
        if imitate_author and not imitate_title:
            dfc['source_text'] = AUTHOR_PROMPT + dfc['author']
        elif not imitate_author and imitate_title:
            dfc['source_text'] = TITLE_PROMPT + dfc['title']
        elif imitate_author and imitate_title:
            dfc['source_text'] = AUTHOR_PROMPT + dfc['author'] + EOS_TOKEN + TITLE_PROMPT + dfc['title']

        dfc['target_text'] = self.data['content']
        dfc = dfc[['source_text', 'target_text']]
        return dfc


def make_train_dataset(path: str = POEM_PATH, split_rate: float = 0.02):
    print(f"Loading data from {path}")
    poem_dataset = PoemDataset(path)
    author_title_poem = poem_dataset.build_dataset(imitate_author=True, imitate_title=False)
    # author_poem = poem_dataset.build_dataset(imitate_author=True, imitate_title=False)
    title_poem = poem_dataset.build_dataset(imitate_author=False, imitate_title=True)
    dataset = pd.concat([
        author_title_poem,
        title_poem,
    ])
    dataset = dataset.sample(frac=1)
    train_df, eval_df = train_test_split(dataset, test_size=split_rate)
    print(f"train: {len(train_df)}, eval: {len(eval_df)}")
    return train_df, eval_df


def train(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_path: str = MODEL_PATH
):
    pl.seed_everything(42)
    model = T5BaseModel()
    model.load_model(model_path, use_gpu=True)
    model.train(
        train_df=train_df,
        eval_df=eval_df,
        source_max_token_len=MAX_IN_TOKENS,
        target_max_token_len=MAX_OUT_TOKENS,
        batch_size=32,
        max_epochs=5,
        use_gpu=True,
        save_dir='./t5-poems'
    )


class Poem():
    def __init__(self, model_path: str) -> None:
        self.model = T5BaseModel()
        self.model.load_model(model_path, use_gpu=False)

    def preprocess(self, author: str = None, title: str = None):
        if author is None or len(author) == 0:
            in_str = TITLE_PROMPT + title
        else:
            in_str = AUTHOR_PROMPT + author + EOS_TOKEN + TITLE_PROMPT + title
        return in_str

    def predict(self, in_str: str):
        return self.model.predict(
            in_str,
            max_length=MAX_OUT_TOKENS,
            num_beams=10,
            do_sample=True
        )


def main():
    model_path = os.path.join(os.getcwd(), "checkpoints/t5-poem")
    model = Poem(model_path)

    while (True):
        author = input(AUTHOR_PROMPT)
        title = input(TITLE_PROMPT)
        s = model.preprocess(author, title)
        print(' ' * 5, s)
        next = model.predict(s)
        print(next[0])


if __name__ == '__main__':
    main()
    # train_df, test_df = make_train_dataset()
    # train(train_df, test_df)
