import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer

from model.data_module import DatasetModule

torch.cuda.empty_cache()

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        model,
        save_dir: str,
        save_only_last_epoch: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output.loss, output.logits

    def loss_cal(self, batch, batch_size):
        input_ids = batch['source_text_input_ids']
        attention_mask = batch['source_text_attention_mask']
        labels_attention_mask = batch['labels_attention_mask']
        labels = batch['labels']

        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = labels_attention_mask
        )
        return loss

    def training_step(self, batch, batch_size):
        """ training step """
        loss = self.loss_cal(batch, batch_size)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        loss = self.loss_cal(batch, batch_size)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        loss = self.loss_cal(batch, batch_size)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.save_dir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )


class T5BaseModel():
    def __init__(self) -> None:
        pass

    def from_pretrained(self, model_name='t5-base') -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    def load_model(self, model_path: str, use_gpu:bool=True):
        """ Load a checkpoint for inference/prediction
        Args:
            model_path (str): path to model directory
            use_gpy (bool): if True, model use gpu for inference/pretraining
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, return_dict=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise "CUDA_NOT_AVAILABLE_EXCEPTION, no gpu found, set use_gpu=False to use CPU"
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        gpus: int = 1,
        save_dir: str = 'outputs',
        early_stopping_patience: int = 0,
        precision=32,
        logger='default',
        dataloader_num_workers: int = 4,
        save_only_last_epoch: bool = False,
    ):
        """train T5 model
        Args:
            train_df (pd.DataFrame): train dataset
            eval_df (pd.DataFrame): eval dataset
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max epochs. Defaults to 5.
            use_gpu (bool, optional): use gpu or not. Defaults to True.
            save_dir (str, optional): output directory. Defaults to 'outputs'.
            early_stopping_patience (int, optional): early stopping patience. Defaults to 0.
            precisions (int, optional): precision. Defaults to 32.
            logger (str, optional): logger. Defaults to 'default'.
            dataloader_num_workers (int, optional): dataloader num workers. Defaults to 4.
            save_only_last_epoch (bool, optional): save only last epoch. Defaults to False.
        """
        self.T5Model = BaseModel(
            tokenizer=self.tokenizer,
            model=self.model,
            save_dir=save_dir,
            save_only_last_epoch=save_only_last_epoch,
        )

        self.dataset = DatasetModule(
            train_df=train_df,
            valid_df=eval_df,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        callbacks = [TQDMProgressBar(refresh_rate=5)]

        if early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor='valid_loss',
                    patience=early_stopping_patience,
                    min_delta=0.00,
                    verbose=True,
                    mode='min'
                ))

        gpus = gpus if use_gpu else 0
        loggers = True if logger == 'default' else logger
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=max_epochs,
            accelerator='gpu' if use_gpu else 'cpu',
            devices=gpus,
            precision=precision,
            log_every_n_steps=1
        )
        trainer.fit(self.T5Model, self.dataset)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ):
        """generate prediction for T5 model
        Args:
            source_text (str): source text
            max_length (int, optional): max length. Defaults to 512.
            num_return_sequences (int, optional): num return sequences. Defaults to 1.
            num_beams (int, optional): num beams. Defaults to 1.
            top_k (int, optional): top k. Defaults to 50.
            top_p (float, optional): top p. Defaults to 0.95.
            do_sample (bool, optional): do sample. Defaults to True.
            repetition_penalty (float, optional): repetition penalty. Defaults to 2.5.
            length_penalty (float, optional): length penalty. Defaults to 1.0.
            early_stopping (bool, optional): early stopping. Defaults to True.
            skip_special_tokens (bool, optional): skip special tokens. Defaults to True.
            clean_up_tokenization_spaces (bool, optional): clean up tokenization spaces. Defaults to True.
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors='pt', add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generate_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences
        )
        preds = [
            self.tokenizer.decoder(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for ids in generate_ids
        ]
        return preds
