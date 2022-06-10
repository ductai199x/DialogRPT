import os
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, SpearmanCorrCoef
from transformers19 import GPT2Model, GPT2Config, GPT2Tokenizer
from dataloader import *


class SimpleScorer(torch.nn.Module):
    def __init__(
        self, 
        pretrained_word_emb: torch.Tensor, 
        pretrained_pos_emb: torch.Tensor,
        seq_len=50,
        hidden_dim=512,
    ):
        # call super class constructor
        super().__init__()

        # Extracting the word vector dimension
        # from our embedding tensor!
        word_dim = pretrained_word_emb.size(1)

        self.word_embedings = torch.nn.Embedding.from_pretrained(pretrained_word_emb, freeze=True)
        self.pos_embedings = torch.nn.Embedding.from_pretrained(pretrained_pos_emb, freeze=True)
        self.pos_ids = torch.arange(0, seq_len).cuda()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(word_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(seq_len),
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.Dropout(0.25),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, 1),
        )

    def forward(self, seq1, seq2):
        # --- your code starts here
        seq1_emb = self.word_embedings(seq1)
        seq2_emb = self.word_embedings(seq2)

        pos_emb = self.pos_embedings(self.pos_ids)
        seq1_emb = seq1_emb + pos_emb
        seq2_emb = seq2_emb + pos_emb

        seq1_score = self.classifier(seq1_emb).mean(dim=1)
        seq2_score = self.classifier(seq2_emb).mean(dim=1)

        return seq1_score.squeeze(), seq2_score.squeeze()


class SimpleScorerPLWrapper(LightningModule):
    def __init__(
        self,
        pretrained_word_emb: torch.Tensor,
        pretrained_pos_emb: torch.Tensor,
        seq_len=50,
        hidden_dim=1024,
        lr=1e-4,
    ):
        super().__init__()
        self.model = SimpleScorer(
            pretrained_word_emb,
            pretrained_pos_emb,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
        )

        self.lr = lr

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.test_spearman_coeff = SpearmanCorrCoef()
        self.example_input_array = (
            torch.zeros((2, 50), dtype=torch.long).to(self.device),
            torch.zeros((2, 50), dtype=torch.long).to(self.device),
        )

    def forward(self, ps, ns):
        return self.model(ps, ns)

    def training_step(self, batch, batch_idx):
        if len(batch["score_pos"]) < 2: return None
        targets = ((batch["score_pos"] - batch["score_neg"]) > 0).long().to(self.device)
        pos_score, neg_score = self(
            batch["pos_samples"], batch["neg_samples"]
        )
        probs = torch.exp(pos_score) / (
            torch.exp(pos_score) + torch.exp(neg_score)
        )

        loss = -torch.log(probs).mean()
        # target_ll = torch.clamp(1 - batch["rank_pos"] - 0.5, min=0.0)
        # loss = (neg_ll - target_ll).mean()

        with torch.no_grad():
            preds = (torch.sigmoid(pos_score) - torch.sigmoid(neg_score)) > 0
        self.train_acc(preds.float(), targets)

        self.log("train_loss", loss, on_epoch=True)
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch["score_pos"]) < 2: return
        targets = ((batch["score_pos"] - batch["score_neg"]) > 0).long().to(self.device)
        pos_score, neg_score = self(
            batch["pos_samples"], batch["neg_samples"]
        )
        probs = torch.exp(pos_score) / (
            torch.exp(pos_score) + torch.exp(neg_score)
        )

        loss = -torch.log(probs).mean()
        # target_ll = torch.clamp(1 - batch["rank_pos"] - 0.5, min=0.0)
        # loss = (neg_ll - target_ll).mean()

        with torch.no_grad():
            preds = (torch.sigmoid(pos_score) - torch.sigmoid(neg_score)) > 0
        self.val_acc(preds.float(), targets)

        self.log("val_loss", loss, on_epoch=True)
        self.log(
            "val_acc",
            self.val_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        if len(batch["score_pos"]) < 2: return
        targets = ((batch["score_pos"] - batch["score_neg"]) > 0).long().to(self.device)
        pos_score, neg_score = self(
            batch["pos_samples"], batch["neg_samples"]
        )
        pos_score = torch.sigmoid(pos_score)
        neg_score = torch.sigmoid(neg_score)
        probs = pos_score - neg_score
        preds = (probs > 0).float()

        self.test_acc(preds, targets)
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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.75)
        reducelr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.7, patience=1)
        # return [optimizer], [steplr]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reducelr,
                "monitor": "val_acc_epoch",
                "frequency": 1,
                "interval": "epoch",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


if __name__ == "__main__":
    feedback = "depth"
    val_ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/test/human_feedback/{feedback}.tsv"
    train_ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/out/{feedback}/train.tsv"
    min_score_gap = 20.0
    min_rank_gap = 0.5
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, max_length=1024, truncation=True)
    val_dl = RedditResponseDataLoader(
        val_ds_path,
        batch_size=256,
        prefetch_batches=64,
        total_num_samples=None,
        need_tokenization=True,
        tokenizer=tokenizer,
        min_score_gap=min_score_gap,
        min_rank_gap=min_rank_gap,
    )
    train_dl = RedditResponseDataLoader(
        train_ds_path,
        batch_size=128,
        prefetch_batches=64,
        total_num_samples=None,
        min_score_gap=min_score_gap,
        min_rank_gap=min_rank_gap,
    )
    # dl = itertools.islice(dl, 99999)

    model_weights = torch.load(f"/media/nas2/Tai/11-reddit-comments-dataset/dialogrpt-model/{feedback}.pth")
    trsf_word_emb = model_weights['transformer.wte.weight'].clone()
    trsf_pos_emb = model_weights['transformer.wpe.weight'].clone()

    max_epochs = 30
    log_dir = "src/lightning_logs"
    model_name = "simple-scorer"
    version = "version_0.3"

    logger = TensorBoardLogger(
        save_dir=log_dir,
        version=version,
        name=model_name,
        log_graph=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_ckpt = ModelCheckpoint(
        dirpath=f"{log_dir}/{model_name}/{version}/checkpoints",
        monitor="val_acc",
        filename=f"{model_name}-{{epoch:02d}}-{{val_acc:.4f}}",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="max",
    )

    # prev_ckpt = "src/lightning_logs/simple-scorer/version_0.2/checkpoints/simple-scorer-epoch=15-val_acc=0.6132.ckpt"
    prev_ckpt = None
    resume = False

    model = SimpleScorerPLWrapper(
        trsf_word_emb,
        trsf_pos_emb,
        hidden_dim=1024,
    )
    if prev_ckpt is not None:
        model = model.load_from_checkpoint(prev_ckpt)

    trainer = Trainer(
        gpus=1,
        max_epochs=max_epochs,
        resume_from_checkpoint=prev_ckpt if resume else None,
        enable_model_summary=True,
        weights_summary="full",
        logger=logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=50), model_ckpt, lr_monitor,
        ],
        fast_dev_run=False,
    )

    try:
        trainer.fit(model, train_dl, val_dl)
        trainer.test(model, val_dl)
    except (Exception) as e:
        raise Exception('Smelly socks').with_traceback(e.__traceback__)
    finally:
        del train_dl, val_dl
