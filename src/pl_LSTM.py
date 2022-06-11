import torch


class LSTMScorer(torch.nn.Module):
    def __init__(
        self,
        pretrained_word_emb: torch.Tensor,
        pretrained_pos_emb: torch.Tensor,
        seq_len=50,
        hidden_dim=512,
        lstm_layers=3,
        word_dropout=0.1,
        pred_dropout=0.1,
        bidirectional=False,
    ):

        super().__init__()
        # Embedding layer
        hidden_dim = 512
        self.num_directions = 2 if bidirectional else 1
        word_dim = pretrained_word_emb.size(1)
        self.word_embedings = torch.nn.Embedding.from_pretrained(pretrained_word_emb, freeze=True)
        self.pos_embedings = torch.nn.Embedding.from_pretrained(pretrained_pos_emb, freeze=True)
        self.pos_ids = torch.arange(0, seq_len).cuda()
        self._embed_dropout = torch.nn.Dropout(word_dropout)
        self._lstm = torch.nn.LSTM(
            word_dim, hidden_dim, num_layers=lstm_layers, bidirectional=bidirectional, batch_first=True
        )
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
