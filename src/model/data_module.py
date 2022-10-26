import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class DataModule(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int,
        target_max_token_len: int,
    ) -> None:
        """ Pytorch Dataset Module for input data
        Args:
            data (pd.DataFrame): Dataframe containing input data
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding input data
            source_max_token_len (int): Max token length for source text
            target_max_token_len (int): Max token length for target text
        """
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        src_text_encoding = self.tokenizer(
            data_row["source_text"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True
        )
        tgt_text_encoding = self.tokenizer(
            data_row['target_text'],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        labels = tgt_text_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            source_text_input_ids=src_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=src_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=tgt_text_encoding["attention_mask"].flatten(),
        )


class DatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.train_dataset = DataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.valid_dataset = DataModule(
            self.valid_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
