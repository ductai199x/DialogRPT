import itertools
from lib2to3.pgen2 import token
import time
from multiprocessing import Event, Lock, Process, Queue, Semaphore, Manager, Value
from queue import Empty as EmptyQueueException
from typing import *

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

column_names = [
    "context",
    "response_pos",
    "response_neg",
    "context_id",
    "response_pos_id",
    "response_neg_id",
    "hour_gap",
    "response_pos_feedback",
    "response_neg_feedback",
    "response_pos_norm_rank",
    "response_neg_norm_rank",
]


# inspired by https://github.com/teddykoker/tinyloader/blob/main/dataloader.py
class RedditResponseDataLoader:
    def __init__(
        self,
        dataset_path,
        batch_size=64,
        num_workers=1,
        prefetch_batches=8,
        total_num_samples: Union[int, None] = None,
        purpose: Literal["gpt", "generic"] = "gpt",
        need_tokenization=False,
        tokenizer=None,
        min_score_gap=0.0,
        min_rank_gap=0.0,
        **kwargs,
    ):
        assert need_tokenization ^ (tokenizer is None)
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.total_num_samples = total_num_samples
        self.dataset_iter = self.get_dataset_iter()

        self.ix_EOS = 50256
        self.ix_OMT = 986
        self.max_seq_length = 50
        self.max_ctxt_length = self.max_seq_length // 2
        self.to_ints = lambda s: list(map(int, s))
        self.get_lengths = lambda seqs: list(map(len, seqs))
        self.pad = lambda seqs: pad_sequence(
            [
                *seqs,
                torch.ones(
                    self.max_seq_length,
                ),
            ],
            batch_first=True,
            padding_value=self.ix_EOS,
        )[:-1]

        self.prefetch_semaphore = Semaphore(prefetch_batches)
        self.output_queue = Queue()
        self.input_lock = Lock()
        self.exit_event = Event()
        self.workers = []

        self.need_tokenization = need_tokenization
        self.tokenizer = tokenizer
        if purpose == "gpt":
            self.prepare_data = self.prepare_data_gpt
        else:
            self.prepare_data = self.prepare_data_generic
        self.min_score_gap = min_score_gap
        self.min_rank_gap = min_rank_gap

    def worker_fn(
        self,
        input_lock: Lock,
        prefetch_smp: Semaphore,
        exit_event: Event,
        output_queue: Queue,
    ):
        while True:
            prefetch_smp.acquire()
            if exit_event.is_set():
                break
            try:
                with input_lock:
                    batch = next(self.dataset_iter)
                output_queue.put(batch)
            except (StopIteration, EOFError):
                exit_event.set()
                break

    def get_dataset_iter(self):
        return pd.read_csv(
            self.dataset_path,
            sep="\t",
            names=column_names,
            iterator=True,
            chunksize=self.batch_size,
            encoding="utf-8",
            engine="c",
            on_bad_lines="warn",
        )

    def __len__(self):
        if self.total_num_samples is not None:
            return -(-self.total_num_samples // self.batch_size)  # ceil without math
        else:
            return None

    def __iter__(self):
        self.dataset_iter.close()
        self.dataset_iter = self.get_dataset_iter()
        self.exit_event.clear()
        if len(self.workers) > 0:
            for w in self.workers:
                w.join(timeout=5.0)
            self.workers.clear()
            self.output_queue._reset()
        for _ in range(self.num_workers):
            worker = Process(
                target=self.worker_fn,
                args=(
                    self.input_lock,
                    self.prefetch_semaphore,
                    self.exit_event,
                    self.output_queue,
                ),
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        return self

    def __next__(self):
        return self.prepare_data(self.get())

    def get(self) -> pd.DataFrame:
        # fetch the data from the output queue
        while True:
            if self.exit_event.is_set():
                raise StopIteration
            try:
                data = self.output_queue.get(timeout=1e-6)
                self.prefetch_semaphore.release()
                break
            except EmptyQueueException:
                continue
        return data

    def prepare_data_generic(self, batch: pd.DataFrame):
        # process the data
        pos_replies = []
        neg_replies = []
        contexts = []
        rpos_lens = []
        rneg_lens = []
        ctxt_lens = []
        for ctxt, rpos, rneg in zip(
            batch["context"].astype("str"),
            batch["response_pos"].astype("str"),
            batch["response_neg"].astype("str"),
        ):
            if self.need_tokenization:
                ctxt = self.tokenizer.encode(ctxt)[-self.max_ctxt_length :]
                rpos = self.tokenizer.encode(rpos)
                rneg = self.tokenizer.encode(rneg)
            else:
                ctxt = np.fromstring(ctxt, dtype=int, sep=" ")[-self.max_ctxt_length :]
                rpos = np.fromstring(rpos, dtype=int, sep=" ")
                rneg = np.fromstring(rneg, dtype=int, sep=" ")

            len_ctxt = len(ctxt)
            pos_reply = rpos[: self.max_seq_length - len_ctxt]
            neg_reply = rneg[: self.max_seq_length - len_ctxt]
            len_rpos = len(pos_reply)
            len_rneg = len(neg_reply)

            pos_replies.append(pos_reply)
            neg_replies.append(neg_reply)
            contexts.append(ctxt)
            rpos_lens.append(len_rpos)
            rneg_lens.append(len_rneg)
            ctxt_lens.append(len_ctxt)

        score_pos = batch["response_pos_feedback"].to_numpy()
        score_neg = batch["response_neg_feedback"].to_numpy()
        rank_pos = batch["response_pos_norm_rank"].to_numpy()
        rank_neg = batch["response_neg_norm_rank"].to_numpy()
        hr_gap = batch["hour_gap"].to_numpy()

        mask_score_gap = (score_pos - score_neg) >= self.min_score_gap
        mask_rank_gap = (rank_pos - rank_neg) >= self.min_rank_gap
        mask = mask_score_gap & mask_rank_gap

        return {
            "pos_replies": pos_replies,
            "neg_replies": neg_replies,
            "contexts": contexts,
            "rpos_lens": rpos_lens,
            "rneg_lens": rneg_lens,
            "ctxt_lens": ctxt_lens,
            "score_pos": score_pos,
            "score_neg": score_neg,
            "rank_pos": rank_pos,
            "rank_neg": rank_neg,
            "hr_gap": hr_gap,
        }

    def prepare_data_gpt(self, batch: pd.DataFrame):
        # process the data
        pos_samples = []
        neg_samples = []
        pos_atn_masks = []
        neg_atn_masks = []
        ctxt_lens = []
        for ctxt, rpos, rneg in zip(
            batch["context"].astype("str"),
            batch["response_pos"].astype("str"),
            batch["response_neg"].astype("str"),
        ):
            if self.need_tokenization:
                ctxt = self.tokenizer.encode(ctxt)[-self.max_ctxt_length :]
                rpos = self.tokenizer.encode(rpos)
                rneg = self.tokenizer.encode(rneg)
            else:
                ctxt = np.fromstring(ctxt, dtype=int, sep=" ")[-self.max_ctxt_length :]
                rpos = np.fromstring(rpos, dtype=int, sep=" ")
                rneg = np.fromstring(rneg, dtype=int, sep=" ")

            pos_sample = [*ctxt, self.ix_EOS, *rpos][: self.max_seq_length]
            neg_sample = [*ctxt, self.ix_EOS, *rneg][: self.max_seq_length]

            len_ctxt = len(ctxt)
            len_rpos = len(pos_sample)
            len_rneg = len(neg_sample)

            pos_sample += [self.ix_EOS] * (self.max_seq_length - len_rpos)
            neg_sample += [self.ix_EOS] * (self.max_seq_length - len_rneg)
            pos_atn_mask = [1] * len_rpos + [0] * (self.max_seq_length - len_rpos)
            neg_atn_mask = [1] * len_rneg + [0] * (self.max_seq_length - len_rneg)

            pos_samples.append(pos_sample)
            neg_samples.append(neg_sample)
            pos_atn_masks.append(pos_atn_mask)
            neg_atn_masks.append(neg_atn_mask)
            ctxt_lens.append(len_ctxt)

        pos_samples = torch.tensor(pos_samples, dtype=torch.long)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        pos_atn_masks = torch.tensor(pos_atn_masks, dtype=torch.long)
        neg_atn_masks = torch.tensor(neg_atn_masks, dtype=torch.long)

        score_pos = torch.tensor(batch["response_pos_feedback"].to_numpy())
        score_neg = torch.tensor(batch["response_neg_feedback"].to_numpy())
        rank_pos = torch.tensor(batch["response_pos_norm_rank"].to_numpy())
        rank_neg = torch.tensor(batch["response_neg_norm_rank"].to_numpy())
        hr_gap = torch.tensor(batch["hour_gap"].to_numpy())

        mask_score_gap = (score_pos - score_neg) > self.min_score_gap
        mask_rank_gap = (rank_pos - rank_neg) > self.min_rank_gap
        mask = mask_score_gap & mask_rank_gap

        return {
            "pos_samples": pos_samples[mask],
            "neg_samples": neg_samples[mask],
            "pos_atn_masks": pos_atn_masks[mask],
            "neg_atn_masks": neg_atn_masks[mask],
            "len_cxt": torch.tensor(ctxt_lens)[mask],
            "score_pos": score_pos[mask],
            "score_neg": score_neg[mask],
            "rank_pos": rank_pos[mask],
            "rank_neg": rank_neg[mask],
            "hr_gap": hr_gap[mask],
        }

    def __del__(self):
        try:
            self.exit_event.set()
            for _ in range(self.num_workers):
                self.prefetch_semaphore.release()

            for w in self.workers:
                w.join(timeout=5.0)

            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                try:
                    w.terminate()
                except:
                    pass
            self.workers.clear()


if __name__ == "__main__":
    feedback = "updown"
    ds_path = f"/home/tai/1-workdir/11-dialog-rpt/data/test/human_feedback/{feedback}.tsv"
    batch_size = 10
    prefetch_batches = 3
    num_workers = 1
    max_iter = 100
    dl = RedditResponseDataLoader(
        ds_path,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
    )
    for epoch in range(3):
        for batch in tqdm(itertools.islice(dl, max_iter), miniters=200):
            time.sleep(1e-2)
        # print(i)
        # print(type(i))
        # break

    # to compare with single process read from pandas
    # df_iter = pd.read_csv(
    #     ds_path,
    #     sep="\t",
    #     names=column_names,
    #     iterator=True,
    #     chunksize=batch_size,
    #     engine="c"
    # )

    # for i in tqdm(itertools.islice(df_iter, max_iter), miniters=200):
    #     dl.prepare_data(i)
    #     time.sleep(1e-2)


# OLD PREPARE DATA - JUST IN CASE NEED LATER!
# ctxt = list(
#     map(lambda s: np.fromstring(s, dtype=int, sep=" ")[-self.max_ctxt_length :], batch["context"].astype("str"))
# )
# rpos = list(map(lambda s: np.fromstring(s, dtype=int, sep=" "), batch["response_pos"].astype("str")))
# rneg = list(map(lambda s: np.fromstring(s, dtype=int, sep=" "), batch["response_neg"].astype("str")))

# pos_samples = list(
#     map(lambda c, r: torch.tensor([*c, self.ix_EOS, *r][: self.max_seq_length]), *[ctxt, rpos])
# )
# neg_samples = list(
#     map(lambda c, r: torch.tensor([*c, self.ix_EOS, *r][: self.max_seq_length]), *[ctxt, rneg])
# )

# len_ctxt = self.get_lengths(ctxt)
# len_rpos = list(map(lambda ,self.get_lengths(pos_samples)))
# len_rneg = self.get_lengths(neg_samples)

# pos_samples = self.pad(pos_samples).long()
# neg_samples = self.pad(neg_samples).long()
