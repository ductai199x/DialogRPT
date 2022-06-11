import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNScorer(torch.nn.Module):
    def __init__(
        self,
        pretrained_word_emb: torch.Tensor,
        pretrained_pos_emb: torch.Tensor,
        seq_len=50,
        hidden_dim=512,
        filter_sizes=[3, 4, 5],
        num_filters=[100, 100, 100],
        num_classes=2,
        dropout=0.5,
        device="cuda",
    ):
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

        super().__init__()
        # Embedding layer

        word_dim = pretrained_word_emb.size(1)
        self.word_embedings = torch.nn.Embedding.from_pretrained(pretrained_word_emb, freeze=True)
        self.pos_embedings = torch.nn.Embedding.from_pretrained(pretrained_pos_emb, freeze=True)
        self.pos_ids = torch.arange(0, seq_len).to(device)
        # Conv Network

        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(in_channels=word_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i])
                for i in range(len(filter_sizes))
            ]
        )
        # Fully-connected layer and Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.ReLU(), nn.Linear(sum(num_filters), num_classes), nn.Linear(2, 1)
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

        x1_pool_list = []
        x2_pool_list = []
        for idx in range(len(x1_conv_list)):
            cd1 = x1_conv_list[idx]
            cd2 = x2_conv_list[idx]

            k1 = cd1.shape[2]
            k2 = cd2.shape[2]

            if torch.is_tensor(k1):
                k1 = k1.tolist()  # Only take the integer not the list

            if torch.is_tensor(k2):
                k2 = k2.tolist()  # Only take the integer not the list

            x1_pool_list.append(F.max_pool1d(cd1, kernel_size=k1))
            x2_pool_list.append(F.max_pool1d(cd2, kernel_size=k2))

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x1_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x1_pool_list], dim=1)

        x2_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x2_pool_list], dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits1 = self.classifier(x1_fc).mean(dim=1)
        logits2 = self.classifier(x2_fc).mean(dim=1)
        return logits1.squeeze(), logits2.squeeze()
