# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                pretrained_word_emb: torch.Tensor, 
                pretrained_pos_emb: torch.Tensor,
                seq_len=50,
                hidden_dim=512,
                filter_sizes=[3, 4, 5],
                num_filters=[100, 100, 100],
                num_classes=2,
                dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer

        word_dim = pretrained_word_emb.size(1)
        self.word_embedings = torch.nn.Embedding.from_pretrained(pretrained_word_emb, freeze=True)
        self.pos_embedings = torch.nn.Embedding.from_pretrained(pretrained_pos_emb, freeze=True)
        self.pos_ids = torch.arange(0, seq_len).cuda()
        # Conv Network

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=word_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(sum(num_filters), num_classes),
            nn.Linear(2, 1)
        )

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

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        
        x1_reshaped = seq1_emb.permute(0, 2, 1)
        x2_reshaped = seq2_emb.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x1_conv_list = [F.relu(conv1d(x1_reshaped)) for conv1d in self.conv1d_list]
        x2_conv_list = [F.relu(conv1d(x2_reshaped)) for conv1d in self.conv1d_list]

        # print('HELLO', x1_conv_list[0].get_shape().as_list()[2])
        # Max pooling. Output shape: (b, num_filters[i], 1)
        
        # x1_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #     for x_conv in x1_conv_list]

        # x2_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #     for x_conv in x2_conv_list]
        x1_pool_list = []
        x2_pool_list = []
        for idx in range(len(x1_conv_list)):
            cd1 = x1_conv_list[idx]
            cd2 = x2_conv_list[idx]

            k1 = cd1.shape[2]
            k2 = cd2.shape[2]

            if torch.is_tensor(k1):
                k1 = k1.tolist() #Only take the integer not the list

            if torch.is_tensor(k2):
                k2 = k2.tolist() #Only take the integer not the list

            x1_pool_list.append(F.max_pool1d(cd1, kernel_size = k1))
            x2_pool_list.append(F.max_pool1d(cd2, kernel_size = k2))
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x1_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x1_pool_list],
                         dim=1)

        x2_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x2_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits1 = self.classifier(x1_fc).mean(dim=1)
        logits2 = self.classifier(x2_fc).mean(dim=1)
        return logits1.squeeze(), logits2.squeeze()

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

feedback = "updown"
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

model = SimpleScorerPLWrapper(
    CNN_NLP,
    trsf_word_emb,
    trsf_pos_emb, 
)


max_epochs = 10
log_dir = "src/lightning_logs"
model_name = feedback
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