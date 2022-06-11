import torch
from transformers19 import GPT2Model, GPT2Config

class RPTScorer(torch.nn.Module):
    def __init__(
        self,
        pretrained_word_emb: torch.Tensor,
        pretrained_pos_emb: torch.Tensor,
        seq_len=50,
        hidden_dim=512,
        **kwargs,
    ):
        super().__init__()
        self.n_embd = 1024
        self.config = GPT2Config(n_embd=self.n_embd, n_layer=24, n_head=16)
        self.transformer = GPT2Model(self.config)
        self.score = torch.nn.Linear(self.n_embd, 1, bias=False)

    def forward(self, pos_samples, pos_atn_masks, neg_samples, neg_atn_masks):
        pos_features, _ = self.transformer(pos_samples, attention_mask=pos_atn_masks)
        neg_features, _ = self.transformer(neg_samples, attention_mask=neg_atn_masks)

        pos_score = self.score(pos_features).squeeze(-1)
        neg_score = self.score(neg_features).squeeze(-1)

        return pos_score.mean(dim=1), neg_score.mean(dim=1)
