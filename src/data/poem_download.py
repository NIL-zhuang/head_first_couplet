import os
import re

import numpy as np
import pandas as pd
from hanziconv import HanziConv
from rich.progress import track

TEST_FLOW = False
MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 10
MAX_CONTENT_CHAR = 64

FILTER_PATTERN = u"\\ï¼.*?\\ï¼|\\ã.*?\\ã|\\{.*?\\}|\\[.*?\\]"

POEM_PATH = os.path.join(os.getcwd(), 'data', 'poems')
POEM_RAW_PATH = os.path.join(POEM_PATH, 'raw')

if TEST_FLOW:
    POEM_DATASET_PATH = os.path.join(POEM_PATH, 'poems_test.csv')
else:
    POEM_DATASET_PATH = os.path.join(POEM_PATH, 'poem.csv')

# https://github.com/chinese-poetry/chinese-poetry
# GitHub - chinese-poetry/chinese-poetry: The most comprehensive database of Chinese poetry
# ð§¶æå¨ä¸­åå¤è¯è¯æ°æ®åº, åå®ä¸¤æè¿ä¸ä¸ååå¤è¯äºº, æ¥è¿5.5ä¸é¦åè¯å 26ä¸å®è¯
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
        filtered_title = re.sub(FILTER_PATTERN, "", row.s_title)
        trim_title = filtered_title[:MAX_TITLE_CHAR].replace(" ", "")
        return trim_title

    def trim_content_fn(row):
        filtered_content = re.sub(FILTER_PATTERN, "", row.s_content)
        # å­å¨ xxx,xxxãï¼xxxï¼ãçæåµï¼æä»¥è¦å»ææåä¸ä¸ªå¥å·
        filtered_content = filtered_content.replace("ãã", "ã").replace("ï¼", "ã")
        trim_content = filtered_content[:MAX_CONTENT_CHAR]
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

    print(f"total poems: {len(poems)}")

    print("removing empty title, content length < MIN_CONTENT_CHAR, empty content")
    empty_title_mask = (poems['s_title_trim'].str.len() == 0)
    too_short_content_mask = (poems['s_content_trim'].str.len() <= MIN_CONTENT_CHAR)
    invalid_mask = (('æ æ­£æ' == poems['s_content_trim']) | ('æ æ­£æ' == poems['s_author_trim']))
    # â¡ to simplified chinese error transform
    invalid_token_mask = ((poems['s_content_trim'].str.contains('â¡')) | poems['s_title_trim'].str.contains('â¡'))
    mask = empty_title_mask | too_short_content_mask | invalid_mask | invalid_token_mask
    filtered_poem = poems.loc[~mask][['s_author_trim', 's_title_trim', 's_content_trim']]
    print(f"filtered poems: {len(filtered_poem)}")

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
