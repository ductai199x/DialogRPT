import pickle
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


def get_test_dataloader(feedback="updown"):
    ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/test/human_feedback/{feedback}.tsv"
    batch_size = 256
    prefetch_batches = min(batch_size // 2, 64)
    if feedback == "updown":
        min_score_gap = 20.0
    elif feedback == "depth":
        min_score_gap = 4.0
    elif feedback == "width":
        min_score_gap = 4.0
    else:
        raise NotImplementedError
    min_rank_gap = 0.5
    return RedditResponseDataLoader(
        ds_path,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        total_num_samples=99999,
        purpose="generic",
        need_tokenization=True,
        decode_after=True,
        tokenizer=tokenizer,
        min_score_gap=min_score_gap,
        min_rank_gap=min_rank_gap,
    )


def get_train_dataloader(feedback="updown", year=2011):
    ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/out/{feedback}/{year}/train.tsv"
    batch_size = 256
    prefetch_batches = min(batch_size // 2, 64)
    # if feedback == "updown":
    #     min_score_gap = 20.0
    # elif feedback == "depth":
    #     min_score_gap = 4.0
    # elif feedback == "width":
    #     min_score_gap = 4.0
    # else:
    #     raise NotImplementedError
    # min_rank_gap = 0.5
    return RedditResponseDataLoader(
        ds_path,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        total_num_samples=None,
        purpose="generic",
        need_tokenization=True,
        decode_after=True,
        tokenizer=tokenizer,
    )


def get_length_baseline():
    test_acc = Accuracy(threshold=0)
    test_spearman_coeff = SpearmanCorrCoef()
    for feedback in ["updown", "width", "depth"]:
        test_dl = get_test_dataloader(feedback)
        test_acc.reset()
        test_spearman_coeff.reset()

        for batch in tqdm(test_dl, desc=f"Length baseline for {feedback}"):
            if len(batch["rpos_lens"]) == 0: continue
            rpos_lens = torch.tensor(batch["rpos_lens"]).float()
            rneg_lens = torch.tensor(batch["rneg_lens"]).float()
            rpos_rank = torch.tensor(batch["rank_pos"]).float()
            rneg_rank = torch.tensor(batch["rank_neg"]).float()

            targets = ((rpos_rank - rneg_rank) > 0).long()
            test_acc.update(rpos_lens - rneg_lens, targets)
            test_spearman_coeff.update(rpos_lens, rpos_rank)
            test_spearman_coeff.update(rneg_lens, rneg_rank)

        print(
            f"{'-'*10} Length results for {feedback} {'-'*10}\n"
            + f"Test acc: {float(test_acc.compute()):.4f}, "
            + f"Test spearman: {float(test_spearman_coeff.compute()):.4f}\n"
        )


# get_length_baseline()


def get_bow_baseline():
    def vectorize(strings):
        """Vectorize the given strings."""
        tf_idf_vectors = tfidf_transform.transform(vectorizer.transform(strings))
        return sp.csr_matrix(tf_idf_vectors, dtype=np.float32, copy=True)

    test_acc = Accuracy(threshold=0)
    test_spearman_coeff = SpearmanCorrCoef()
    for feedback in ["updown", "width", "depth"]:

        test_acc.reset()
        test_spearman_coeff.reset()

        # We gotta first fit the TfidfTransformer.
        vectorizer = HashingVectorizer()
        tfidf_transform = TfidfTransformer()

        train_dl = get_train_dataloader(feedback, year=2011)
        for batch in tqdm(train_dl, desc=f"Fitting TfidfTransformer for {feedback}"):
            pos_replies = batch["pos_replies"]
            neg_replies = batch["neg_replies"]
            contexts = batch["contexts"]
            tfidf_transform.fit(vectorizer.fit_transform(contexts + pos_replies + neg_replies))

        train_dl = get_train_dataloader(feedback, year=2012)
        for batch in tqdm(train_dl, desc=f"Fitting TfidfTransformer for {feedback}"):
            pos_replies = batch["pos_replies"]
            neg_replies = batch["neg_replies"]
            contexts = batch["contexts"]
            tfidf_transform.fit(vectorizer.fit_transform(contexts + pos_replies + neg_replies))

        test_dl = get_test_dataloader(feedback)
        pbar = tqdm(total=len(test_dl), desc=f"BoW baseline for {feedback}")
        for batch in test_dl:
            rpos_rank = torch.tensor(batch["rank_pos"]).float()
            rneg_rank = torch.tensor(batch["rank_neg"]).float()
            pos_replies = batch["pos_replies"]
            neg_replies = batch["neg_replies"]
            contexts = batch["contexts"]

            preds = []
            rpos_score = []
            rneg_score = []
            for ctxt, rpos, rneg in zip(contexts, pos_replies, neg_replies):
                ctxt_mat = vectorize([ctxt])
                resp_mat = vectorize([rpos, rneg])
                sim_mat = ctxt_mat.dot(resp_mat.T).toarray()
                pred = np.argmax(sim_mat, axis=1)[0]

                preds.append(1 - pred)
                rpos_score.append(sim_mat[0, 0])
                rneg_score.append(sim_mat[0, 1])

            targets = ((rpos_rank - rneg_rank) > 0).long()
            test_acc.update(torch.tensor(preds), targets)
            test_spearman_coeff.update(torch.tensor(rpos_score), rpos_rank)
            test_spearman_coeff.update(torch.tensor(rneg_score), rneg_rank)

            pbar.set_postfix({"acc": float(test_acc.compute()), "spearman": float(test_spearman_coeff.compute())})
            pbar.update()

        print(
                f"{'-'*10} BoW results for {feedback} {'-'*10}\n"
                + f"Test acc: {float(test_acc.compute()):.4f}, "
                + f"Test spearman: {float(test_spearman_coeff.compute()):.4f}\n"
            )
        
    # return vectorizer, tfidf_transform


get_bow_baseline()
