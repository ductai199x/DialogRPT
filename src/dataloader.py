import itertools
import time
from multiprocessing import Event, Lock, Process, Queue, Semaphore, Manager, Value
from queue import Empty as EmptyQueueException

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
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
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
        for _ in range(num_workers):
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
                return
            try:
                with input_lock:
                    batch = next(self.dataset_iter)
                output_queue.put(self.prepare_data(batch))
            except StopIteration:
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

    def __iter__(self):
        self.dataset_iter = self.get_dataset_iter()
        self.exit_event.clear()
        return self

    def __next__(self):
        return self.get()

    def get(self) -> pd.DataFrame:
        # fetch the data from the output queue
        while True:
            try:
                data = self.output_queue.get(timeout=1e-6)
                self.prefetch_semaphore.release()
                break
            except EmptyQueueException:
                continue
        return data

    def prepare_data(self, batch: pd.DataFrame):
        # process the data
        ctxt = list(
            map(lambda s: np.fromstring(s, dtype=int, sep=" ")[-self.max_ctxt_length :], batch["context"].astype("str"))
        )
        rpos = list(map(lambda s: np.fromstring(s, dtype=int, sep=" "), batch["response_pos"].astype("str")))
        rneg = list(map(lambda s: np.fromstring(s, dtype=int, sep=" "), batch["response_neg"].astype("str")))

        pos_samples = list(
            map(lambda c, r: torch.tensor([*c, self.ix_EOS, *r][: self.max_seq_length]), *[ctxt, rpos])
        )
        neg_samples = list(
            map(lambda c, r: torch.tensor([*c, self.ix_EOS, *r][: self.max_seq_length]), *[ctxt, rneg])
        )

        len_ctxt = self.get_lengths(ctxt)
        len_rpos = self.get_lengths(pos_samples)
        len_rneg = self.get_lengths(neg_samples)

        pos_samples = self.pad(pos_samples).long()
        neg_samples = self.pad(neg_samples).long()

        score_pos = batch["response_pos_feedback"].to_list()
        score_neg = batch["response_neg_feedback"].to_list()
        rank_pos = batch["response_pos_norm_rank"].to_list()
        rank_neg = batch["response_neg_norm_rank"].to_list()
        hr_gap = batch["hour_gap"].to_list()

        return {
            "ids_pos": pos_samples,
            "ids_neg": neg_samples,
            "len_pos": len_rpos,
            "len_neg": len_rneg,
            "len_cxt": len_ctxt,
            "score_pos": score_pos,
            "score_neg": score_neg,
            "rank_pos": rank_pos,
            "rank_neg": rank_neg,
            "hr_gap": hr_gap,
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
                if w.is_alive():  # manually terminate worker if all else fails
                    w.terminate()


if __name__ == "__main__":
    ds_path = "data/out/updown/2012/train.tsv"
    batch_size = 64
    prefetch_batches = 15
    num_workers = 1
    max_iter = 10_000
    dl = RedditResponseDataLoader(
        ds_path,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
    )
    for i in tqdm(itertools.islice(dl, max_iter), miniters=200):
        time.sleep(1e-2)
        # print(i)
        # print(type(i))
        # break

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
        
