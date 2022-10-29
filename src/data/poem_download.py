import os

import numpy as np
import pandas as pd
from hanziconv import HanziConv
from rich.progress import track
from tqdm.notebook import tqdm

TEST_FLOW = True
MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 10
MAX_CONTENT_CHAR = 64

POEM_PATH = os.path.join(os.getcwd(), 'data', 'poems')
POEM_RAW_PATH = os.path.join(POEM_PATH, 'raw')

if TEST_FLOW:
    POEM_DATASET_PATH = os.path.join(POEM_PATH, 'poems_test.csv')
else:
    POEM_DATASET_PATH = os.path.join(POEM_PATH, 'poem.csv')

# https://github.com/chinese-poetry/chinese-poetry
# GitHub - chinese-poetry/chinese-poetry: The most comprehensive database of Chinese poetry
# 🧶最全中华古诗词数据库, 唐宋两朝近一万四千古诗人, 接近5.5万首唐诗加26万宋诗
POEM_CONTENT = {
    'tang': {
        'total': 58,
    },
    'song': {
        'total': 255,
    }
}


def get_poems(is_test=True):
    df_list = []
    for dynasty in POEM_CONTENT:
        size = 3 if is_test else POEM_CONTENT[dynasty]['total']
        for i in track(range(size), description=f"Loading {dynasty}"):
            fpath = os.path.join(POEM_RAW_PATH, f"poet.{dynasty}.{i*1000}.json")
            df_list.append(pd.read_json(fpath))
    return pd.concat(df_list)


def make_poem_dataset(raw_poem: pd.DataFrame):

    def convert_simple_chinese(t_chinese: str):
        return HanziConv.toSimplified(t_chinese)

    def trim_author_fn(row):
        return row.s_author[:MAX_AUTHOR_CHAR]

    def trim_title_fn(row):
        trim_title = row.s_title[:MAX_TITLE_CHAR].replace(" ", "").replace("(", "").replace(")", "")
        return trim_title

    def trim_content_fn(row):
        trim_content = row.s_content[:MAX_CONTENT_CHAR]
        return trim_content

    poems = raw_poem.copy()
    poems['concat_paragraphs'] = [''.join(map(str, l)) for l in poems['paragraphs']]
    poems = poems[['author', 'title', 'concat_paragraphs']]

    print("changing from traditional to simplified chinese")
    poems['s_content'] = poems.apply(lambda row: convert_simple_chinese(''.join(row.concat_paragraphs)), axis=1)
    poems['s_title'] = poems.apply(lambda row: convert_simple_chinese(''.join(row.title)), axis=1)
    poems['s_author'] = poems.apply(lambda row: convert_simple_chinese(''.join(row.author)), axis=1)

    print("trim author, title, content from too long")
    poems['s_author_trim'] = poems.apply(trim_author_fn, axis=1)
    poems['s_title_trim'] = poems.apply(trim_title_fn, axis=1)
    poems['s_content_trim'] = poems.apply(trim_content_fn, axis=1)

    print("removing empty title, content length < MIN_CONTENT_CHAR, empty content")
    empty_title_mask = (poems['s_title_trim'].str.len() == 0)
    too_short_content_mask = (poems['s_content_trim'].str.len() <= MIN_CONTENT_CHAR)
    invalid_mask = (('无正文' == poems['s_content_trim']) | ('无正文' == poems['s_author_trim']))
    too_short_mask = empty_title_mask | too_short_content_mask | invalid_mask
    filtered_poem = poems.loc[~too_short_mask][['s_author_trim', 's_title_trim', 's_content_trim']]

    # get quality dataset
    filtered_poem.rename(columns={'s_author_trim': 'author', 's_title_trim': 'title', 's_content_trim': 'content'}, inplace=True)
    return filtered_poem


def main():
    raw_poem = get_poems(is_test=TEST_FLOW)
    filtered_poems = make_poem_dataset(raw_poem)

    print(filtered_poems.sample(3))
    print(filtered_poems.describe())

    filtered_poems.to_csv(POEM_DATASET_PATH, index=False)


if __name__ == '__main__':
    main()
