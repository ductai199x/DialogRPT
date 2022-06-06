import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from torchmetrics import Accuracy, SpearmanCorrCoef
from tqdm.auto import tqdm

from dataloader import *
from pl_train import *
from transformers19 import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, max_length=1024, truncation=True)


def get_dataloader(feedback="updown"):
    ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/test/human_feedback/{feedback}.tsv"
    batch_size = 256
    prefetch_batches = min(batch_size // 2, 64)
    num_workers = 1
    return RedditResponseDataLoader(
        ds_path,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        total_num_samples=99999,
        purpose="generic",
    )


def get_length_baseline():
    for feedback in ["updown", "width", "depth"]:
        dl = get_dataloader(feedback)
        test_acc = Accuracy(threshold=0)
        test_spearman_coeff = SpearmanCorrCoef()

        test_acc.reset()
        test_spearman_coeff.reset()

        for batch in tqdm(dl, desc=f"Length baseline for {feedback}"):
            rpos_lens = batch["rpos_lens"]
            rneg_lens = batch["rneg_lens"]
            len_diff = torch.tensor([p - n for p, n in zip(rpos_lens, rneg_lens)]).long()
            score_diff = torch.tensor(batch["score_pos"] - batch["score_neg"]).float()
            targets = (score_diff > 0).long()
            test_acc.update(len_diff.float(), targets)
            test_spearman_coeff.update(len_diff.float(), score_diff)

        print(
            f"Test acc: {float(test_acc.compute()):.4f}, Test spearman: {float(test_spearman_coeff.compute()):.4f}"
        )


get_length_baseline()


def get_bow_baseline():
    def vectorize(strings):
        """Vectorize the given strings."""
        tf_idf_vectors = tfidf_transform.transform(vectorizer.transform(strings))
        return sp.csr_matrix(tf_idf_vectors, dtype=np.float32, copy=True)

    # for feedback in ["updown", "width", "depth"]:
    feedback = "updown"
    test_acc = Accuracy(threshold=0)
    test_spearman_coeff = SpearmanCorrCoef()

    test_acc.reset()
    test_spearman_coeff.reset()

    # We gotta first fit the TfidfTransformer.
    vectorizer = HashingVectorizer()
    tfidf_transform = TfidfTransformer()

    train_dl = get_dataloader(feedback, year=2011, type="train")
    for batch in tqdm(train_dl, desc=f"Fitting TfidfTransformer for {feedback}"):
        pos_replies = list(map(tokenizer.decode, batch["pos_replies"]))
        neg_replies = list(map(tokenizer.decode, batch["neg_replies"]))
        contexts = list(map(tokenizer.decode, batch["contexts"]))
        tfidf_transform.fit(vectorizer.transform(contexts + pos_replies + neg_replies))

    train_dl = get_dataloader(feedback, year=2012, type="train")
    for batch in tqdm(train_dl, desc=f"Fitting TfidfTransformer for {feedback}"):
        pos_replies = list(map(tokenizer.decode, batch["pos_replies"]))
        neg_replies = list(map(tokenizer.decode, batch["neg_replies"]))
        contexts = list(map(tokenizer.decode, batch["contexts"]))
        tfidf_transform.fit(vectorizer.transform(contexts + pos_replies + neg_replies))

    test_dl = get_dataloader(feedback, year=2013, type="vali")
    pbar = tqdm(total=len(test_dl), desc=f"BoW baseline for {feedback}")
    for batch in test_dl:
        pos_replies = list(map(tokenizer.decode, batch["pos_replies"]))
        neg_replies = list(map(tokenizer.decode, batch["neg_replies"]))
        contexts = list(map(tokenizer.decode, batch["contexts"]))

        preds = []
        scores = []
        for ctxt, rpos, rneg in zip(contexts, pos_replies, neg_replies):
            ctxt_mat = vectorize([ctxt])
            resp_mat = vectorize([rpos, rneg])
            sim_mat = ctxt_mat.dot(resp_mat.T).toarray()
            pred = np.argmax(sim_mat, axis=1)[0]
            score = sim_mat[0, pred]

            preds.append(1 - pred)
            scores.append(score)

        true_score_diff = torch.tensor(batch["score_pos"] - batch["score_neg"]).float()
        targets = (true_score_diff > 0).long()
        test_acc.update(torch.tensor(preds), targets)
        test_spearman_coeff.update(torch.tensor(scores), true_score_diff)

        pbar.set_postfix({"acc": float(test_acc.compute())})
        pbar.update()

    print(
        f"Test acc: {float(test_acc.compute()):.4f}, Test spearman: {float(test_spearman_coeff.compute()):.4f}"
    )
    return vectorizer, tfidf_transform


vectorizer, tfidf_transform = get_bow_baseline()
