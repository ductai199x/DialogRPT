# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                pretrained_word_emb: torch.Tensor, 
                pretrained_pos_emb: torch.Tensor,
                seq_len=50,
                hidden_dim=512,
                lstm_layers = 3,
                word_dropout = 0.1, 
                pred_dropout = 0.1,
                bidirectional = False
                ):

        super(LSTM, self).__init__()
        # Embedding layer
        hidden_dim = 512
        self.num_directions = 2 if bidirectional else 1
        word_dim = pretrained_word_emb.size(1)
        self.word_embedings = torch.nn.Embedding.from_pretrained(pretrained_word_emb, freeze=True)
        self.pos_embedings = torch.nn.Embedding.from_pretrained(pretrained_pos_emb, freeze=True)
        self.pos_ids = torch.arange(0, seq_len).cuda()
        self._embed_dropout = torch.nn.Dropout(word_dropout)
        self._lstm = torch.nn.LSTM(word_dim, hidden_dim, num_layers = lstm_layers, bidirectional = bidirectional, batch_first=True)
        self._ReLU = torch.nn.ReLU()
        self._pred_dropout = torch.nn.Dropout(pred_dropout)
        self._pred = torch.nn.Linear(hidden_dim, 1, bias=False)



    def forward(self, seq1, seq2):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        seq1_emb = self.word_embedings(seq1).float()
        seq2_emb = self.word_embedings(seq2).float()

        pos_emb = self.pos_embedings(self.pos_ids).float()
        seq1_emb = seq1_emb + pos_emb
        seq2_emb = seq2_emb + pos_emb

        seq1_emb = self._embed_dropout(seq1_emb)
        seq2_emb = self._embed_dropout(seq2_emb)

        s1, _ = self._lstm(seq1_emb)
        s2, _ = self._lstm(seq2_emb)

        s1 = self._ReLU(s1)
        s2 = self._ReLU(s2)

        s2 = self._pred_dropout(s2)
        s1 = self._pred_dropout(s1)

        s1 = self._pred(s1).mean(dim=1)
        s2 = self._pred(s2).mean(dim=1)

        return s1.squeeze(), s2.squeeze()

# %%
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
from pl_train_simple_scorer import SimpleScorerPLWrapper, SimpleScorer

mname = "LSTM"
feedback = "width"
val_ds_path = f"/home/ec2-user/DialogRPT/data/test/human_feedback/{feedback}.tsv"
train_ds_path = f"/home/ec2-user/DialogRPT/data/out/{feedback}/train.tsv"

if feedback == "updown":
    min_score_gap = 20.0
else:
    min_score_gap = 4.0

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
    min_rank_gap=min_rank_gap
    )
    
train_dl = RedditResponseDataLoader(
    train_ds_path,
    batch_size=164,
    prefetch_batches=64,
    total_num_samples=None,
    min_score_gap=min_score_gap,
    min_rank_gap=min_rank_gap,
)
    # dl = itertools.islice(dl, 99999)

model_weights = torch.load(f"/home/ec2-user/DialogRPT/restore/{feedback}.pth")
trsf_word_emb = model_weights['transformer.wte.weight'].clone()
trsf_pos_emb = model_weights['transformer.wpe.weight'].clone()

#prev_ckpt = "/home/ec2-user/DialogRPT/src/lightning_logs/depth/version_0/checkpoints/depth-epoch=09-val_acc=0.6243.ckpt"

prev_ckpt = None

resume = False

if prev_ckpt is not None:
    model = SimpleScorerPLWrapper.load_from_checkpoint(
        prev_ckpt,
        model=LSTM,
        pretrained_word_emb=trsf_word_emb,
        pretrained_pos_emb=trsf_pos_emb,
        hidden_dim=1024,
    )

else:
    model = SimpleScorerPLWrapper(
        LSTM,
        trsf_word_emb,
        trsf_pos_emb,
    )

max_epochs = 10
log_dir = "src/lightning_logs"
model_name = mname+feedback
version = "version_0"

logger = TensorBoardLogger(
    save_dir=log_dir,
    version=version,
    name=model_name,
    log_graph=True,
)

model_ckpt = ModelCheckpoint(
    dirpath=f"{log_dir}/{model_name}/{version}/checkpoints",
    monitor="val_acc",
    filename=f"{model_name}-{{epoch:02d}}-{{val_acc:.4f}}",
    verbose=True,
    save_last=True,
    save_top_k=1,
    mode="max",
)

trainer = Trainer(
    gpus=1,
    max_epochs=max_epochs,
    resume_from_checkpoint=None,
    enable_model_summary=True,
    logger=logger,
    callbacks=[
        TQDMProgressBar(refresh_rate=50), model_ckpt,
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