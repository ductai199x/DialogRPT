import torch


class FullyConnectedScorer(torch.nn.Module):
    def __init__(
        self,
        pretrained_word_emb: torch.Tensor,
        pretrained_pos_emb: torch.Tensor,
        seq_len=50,
        hidden_dim=1024,
    ):
        super().__init__()

        word_dim = pretrained_word_emb.size(1)

        self.word_embedings = torch.nn.Embedding.from_pretrained(pretrained_word_emb, freeze=False)
        self.pos_embedings = torch.nn.Embedding.from_pretrained(pretrained_pos_emb, freeze=False)
        self.pos_ids = torch.arange(0, seq_len).cuda()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(word_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(seq_len),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Dropout(0.25),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq1, seq2):
        seq1_emb = self.word_embedings(seq1)
        seq2_emb = self.word_embedings(seq2)

        pos_emb = self.pos_embedings(self.pos_ids)
        seq1_emb = seq1_emb + pos_emb
        seq2_emb = seq2_emb + pos_emb

        seq1_score = self.classifier(seq1_emb).mean(dim=1)
        seq2_score = self.classifier(seq2_emb).mean(dim=1)

        return seq1_score.squeeze(), seq2_score.squeeze()
