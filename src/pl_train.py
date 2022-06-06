import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy, SpearmanCorrCoef
from transformers19 import GPT2Model, GPT2Config, GPT2Tokenizer
from dataloader import *


class Scorer(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.n_embd = 1024
        self.config = GPT2Config(n_embd=self.n_embd, n_layer=24, n_head=16)
        self.transformer = GPT2Model(self.config)
        self.score = torch.nn.Linear(self.n_embd, 1, bias=False)

        self.ix_EOS = 50256
        self.ix_OMT = 986

    def forward(self, pos_samples, pos_atn_masks, neg_samples, neg_atn_masks):
        pos_features, _ = self.transformer(pos_samples, attention_mask=pos_atn_masks)
        neg_features, _ = self.transformer(neg_samples, attention_mask=neg_atn_masks)

        pos_score = self.score(pos_features).squeeze(-1)
        neg_score = self.score(neg_features).squeeze(-1)

        pos_score = torch.stack(
            [
                pos_score[i, pos_atn_masks[i].sum() - 1]
                for i in range(pos_samples.shape[0])
            ]
        )
        neg_score = torch.stack(
            [
                neg_score[i, neg_atn_masks[i].sum() - 1]
                for i in range(neg_samples.shape[0])
            ]
        )

        return pos_score, neg_score


class ScorerPLWrapper(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Scorer()

        self.lr = 3e-5

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.test_spearman_coeff = SpearmanCorrCoef()
        self.example_input_array = (
            torch.zeros((1, 50), dtype=torch.long).to(self.device),
            torch.zeros((1, 50), dtype=torch.long).to(self.device),
            torch.zeros((1, 50), dtype=torch.long).to(self.device),
            torch.zeros((1, 50), dtype=torch.long).to(self.device),
        )

    def forward(self, ps, pm, ns, nm):
        return self.model(ps, pm, ns, nm)

    def training_step(self, batch, batch_idx):
        targets = ((batch["score_pos"] - batch["score_neg"]) > 0).long().to(self.device)
        pos_score, neg_score = self(
            batch["pos_samples"], batch["pos_atn_masks"], batch["neg_samples"], batch["neg_atn_masks"]
        )
        probs = torch.exp(pos_score) / (
            torch.exp(pos_score) + torch.exp(neg_score)
        )
        # probs = torch.exp(pos_score[:, [0, -1]]) / (
        #     torch.exp(pos_score[:, [0, -1]]) + torch.exp(neg_score[:, [0, -1]])
        # )
        probs = probs.mean(dim=1)
        neg_ll = -torch.log(probs)
        target_ll = torch.clamp(1 - batch["rank_pos"] - 0.5, min=0.0)
        loss = (neg_ll - target_ll).mean()

        self.log("train_loss", loss)
        self.train_acc(probs, targets)
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        targets = ((batch["score_pos"] - batch["score_neg"]) > 0).long().to(self.device)
        pos_score, neg_score = self(
            batch["pos_samples"], batch["pos_atn_masks"], batch["neg_samples"], batch["neg_atn_masks"]
        )
        probs = torch.exp(pos_score) / (
            torch.exp(pos_score) + torch.exp(neg_score)
        )
        # probs = torch.exp(pos_score[:, [0, -1]]) / (
        #     torch.exp(pos_score[:, [0, -1]]) + torch.exp(neg_score[:, [0, -1]])
        # )
        probs = probs.mean(dim=1)
        neg_ll = -torch.log(probs)
        target_ll = torch.clamp(1 - batch["rank_pos"] - 0.5, min=0.0)
        loss = (neg_ll - target_ll).mean()

        self.log("val_loss", loss)
        self.val_acc(probs, targets)
        self.log(
            "val_acc",
            self.val_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        targets = ((batch["score_pos"] - batch["score_neg"]) > 0).long().to(self.device)
        pos_score, neg_score = self(
            batch["pos_samples"], batch["pos_atn_masks"], batch["neg_samples"], batch["neg_atn_masks"]
        )
        pos_score = torch.sigmoid(pos_score)
        neg_score = torch.sigmoid(neg_score)
        probs = pos_score - neg_score

        # probs = torch.exp(pos_score) / (
        #     torch.exp(pos_score) + torch.exp(neg_score)
        # )
        # probs = torch.exp(pos_score[:, [0, -1]]) / (
        #     torch.exp(pos_score[:, [0, -1]]) + torch.exp(neg_score[:, [0, -1]])
        # )
        # probs = probs.mean(dim=1)
        # neg_ll = -torch.log(probs)
        # target_ll = torch.clamp(1 - batch["rank_pos"] - 0.5, min=0.0)
        # loss = (neg_ll - target_ll).mean()

        # self.log("test_loss", probs, on_epoch=True)
        self.test_acc((probs > 0).float(), targets)
        self.log(
            "test_acc",
            self.test_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.test_spearman_coeff(pos_score, (batch["rank_pos"]).float())
        self.test_spearman_coeff(neg_score, (batch["rank_neg"]).float())
        self.log(
            "test_spearman_coeff",
            self.test_spearman_coeff,
            on_epoch=True,
        )

    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()

    def on_test_epoch_start(self) -> None:
        self.test_acc.reset()
        self.test_spearman_coeff.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]


if __name__ == "__main__":
    feedback = "updown"
    ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/test/human_feedback/{feedback}.tsv"
    batch_size = 256
    prefetch_batches = min(batch_size // 2, 64)
    min_score_gap = 20.0
    min_rank_gap = 0.5
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, max_length=1024, truncation=True)
    dl = RedditResponseDataLoader(
        ds_path,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        total_num_samples=99999,
        need_tokenization=True,
        tokenizer=tokenizer,
        min_score_gap=min_score_gap,
        min_rank_gap=min_rank_gap,
    )
    # dl = itertools.islice(dl, 99999)

    model = ScorerPLWrapper()
    # model_weights = torch.load(f"/media/nas2/Tai/11-reddit-comments-dataset/dialogrpt-model/{feedback}.pth")
    # model.model.load_state_dict(model_weights)

    logger = pl.loggers.TensorBoardLogger(
        save_dir="src/lightning_logs",
        version=f"version_1",
        name="gpt-2-scorer",
        log_graph=True,
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=1,
        resume_from_checkpoint=None,
        enable_model_summary=True,
        logger=logger,
        callbacks=[
            pl.callbacks.TQDMProgressBar(refresh_rate=1),
        ],
        fast_dev_run=False,
        limit_train_batches=1000,
    )

    # trainer.fit(model, dl)
    trainer.test(model, dl)
