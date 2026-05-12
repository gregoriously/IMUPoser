import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import DataLoader

BATCH_SIZE = 32


class IMUPoserDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_pct=0.9):
        super().__init__()
        self.dataset_name = dataset.__name__
        self.dataset = dataset
        self.batch_size = BATCH_SIZE
        self.train_pct = train_pct
        self.save_hyperparameters(ignore=["dataset"])  # confirm what this will do

    def setup(
        self, stage=None
    ):  # this may fail as validation stage doesn't get accessed...
        if stage == "fit":
            full_train_dataset = self.dataset(split="train")
            train_size = int(self.train_pct * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train_dataset,
                [train_size, val_size],
            )
        elif stage == "test":
            self.test_dataset = self.dataset(split="test")

    @staticmethod
    def pad_seq(batch):
        inputs = [item[0] for item in batch]
        outputs = [item[1] for item in batch]

        input_lens = [item.shape[0] for item in inputs]
        output_lens = [item.shape[0] for item in outputs]

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        outputs = nn.utils.rnn.pad_sequence(outputs, batch_first=True)
        return inputs, outputs, input_lens, output_lens

    def get_data_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.pad_seq,
            num_workers=8,
            shuffle=shuffle,
            # use pin memory? could be free performance
        )

    def train_dataloader(self):
        return self.get_data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.get_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self.get_data_loader(self.test_dataset)

    def predict_dataloader(self):
        return self.get_data_loader(self.test_dataset)
