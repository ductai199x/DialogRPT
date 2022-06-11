import argparse
import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import *
from pl_RPT import RPTScorer
from pl_CNN import CNNScorer
from pl_FC import FullyConnectedScorer
from pl_LSTM import LSTMScorer
from pl_scorer_wrapper import ScorerPLWrapper
from shared import download_model
from transformers19 import GPT2Tokenizer

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()


def get_testing_data() -> RedditResponseDataLoader:
    val_ds_path = os.path.abspath(f"{ARGS.rootdir}/data/test/human_feedback/{ARGS.feedback}.tsv")
    if ARGS.feedback == "updown":
        min_score_gap = 20.0
    elif ARGS.feedback in ["depth", "width"]:
        min_score_gap = 4.0
    min_rank_gap = 0.5
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, max_length=1024, truncation=True)
    return RedditResponseDataLoader(
        val_ds_path,
        batch_size=256,
        prefetch_batches=64,
        total_num_samples=None,
        need_tokenization=True,
        tokenizer=tokenizer,
        min_score_gap=min_score_gap,
        min_rank_gap=min_rank_gap,
    )


def get_training_data() -> RedditResponseDataLoader:
    train_ds_path = os.path.abspath(f"{ARGS.rootdir}/data/out/{ARGS.feedback}/train.tsv")
    if ARGS.feedback == "updown":
        min_score_gap = 20.0
    elif ARGS.feedback in ["depth", "width"]:
        min_score_gap = 4.0
    min_rank_gap = 0.5
    return RedditResponseDataLoader(
        train_ds_path,
        batch_size=128,
        prefetch_batches=64,
        total_num_samples=None,
        min_score_gap=min_score_gap,
        min_rank_gap=min_rank_gap,
    )


def get_model() -> LightningModule:
    weights_path = os.path.abspath(f"{ARGS.rootdir}/restore/{ARGS.feedback}.pth")
    if not os.path.exists(weights_path):
        download_model(f"{ARGS.rootdir}/restore", False)
    model_weights = torch.load(weights_path, map_location="cpu" if ARGS.cpu else "cuda")
    trsf_word_emb = model_weights['transformer.wte.weight'].clone()
    trsf_pos_emb = model_weights['transformer.wpe.weight'].clone()

    prev_ckpt = (f"{ARGS.rootdir}/lightning_checkpoints/{ARGS.arch}-scorer/"
                + f"{ARGS.feedback}/checkpoints/best.ckpt")
    
    if ARGS.arch == "FC-GPT":
        model_class = FullyConnectedScorer
    elif ARGS.arch == "FC-GLOVE":
        raise NotImplementedError
    elif ARGS.arch == "CNN":
        model_class = CNNScorer
    elif ARGS.arch == "LSTM":
        model_class = LSTMScorer
    elif ARGS.arch == "RPT":
        model_class = RPTScorer
    else:
        raise NotImplementedError

    configs = {
        "model": model_class,
        "pretrained_word_emb": trsf_word_emb,
        "pretrained_pos_emb": trsf_pos_emb,
        "hidden_dim": 1024,
        "device": "cpu" if ARGS.cpu else "cuda",
    }

    if ARGS.arch == "RPT":
        scorer = ScorerPLWrapper(**configs)
        scorer.model.load_state_dict(model_weights)
        return scorer
    else:
        return ScorerPLWrapper.load_from_checkpoint(
            prev_ckpt,
            **configs
        )


def train():
    train_dl = get_training_data()
    val_dl = get_testing_data()
    model = get_model()

    log_dir = "lightning_logs"
    model_name = f"{ARGS.arch}-scorer"
    version = f"{ARGS.feedback}_v0"

    logger = TensorBoardLogger(
        save_dir=log_dir,
        version=version,
        name=model_name,
        log_graph=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
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
        gpus=0 if ARGS.cpu else 1,
        max_epochs=ARGS.max_epoch,
        enable_model_summary=True,
        logger=logger,
        weights_summary="full",
        callbacks=[
            TQDMProgressBar(refresh_rate=50),
            lr_monitor,
            model_ckpt,
        ],
        fast_dev_run=False,
    )
    try:
        trainer.fit(model, train_dl, val_dl)
    except (Exception) as e:
        print(e)
        raise Exception('Smelly socks').with_traceback(e.__traceback__)
    finally:
        del val_dl


def evaluate():
    val_dl = get_testing_data()
    model = get_model()
    trainer = Trainer(
        gpus=0 if ARGS.cpu else 1,
        enable_model_summary=True,
        weights_summary="full",
        logger=None,
        callbacks=[
            TQDMProgressBar(refresh_rate=50),
        ],
        fast_dev_run=False,
    )

    try:
        trainer.test(model, val_dl)
    except (Exception) as e:
        print(e)
        raise Exception('Smelly socks').with_traceback(e.__traceback__)
    finally:
        del val_dl


def predict():
    ix_EOS = 50256
    ix_OMT = 986
    max_seq_length = 50
    max_ctxt_length = max_seq_length // 2
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, max_length=1024, truncation=True)
    ctxt = tokenizer.encode(ARGS.context)[-max_ctxt_length :]
    seq1 = tokenizer.encode(ARGS.seq1)
    seq2 = tokenizer.encode(ARGS.seq2)

    sample1 = [*ctxt, ix_EOS, *seq1][: max_seq_length]
    sample2 = [*ctxt, ix_EOS, *seq2][: max_seq_length]

    len_ctxt = len(ctxt)
    len_seq1 = len(sample1)
    len_seq2 = len(sample2)

    sample1 += [ix_EOS] * (max_seq_length - len_seq1)
    sample2 += [ix_EOS] * (max_seq_length - len_seq2)
    
    sample1_atn_mask = [1] * len_seq1 + [0] * (max_seq_length - len_seq1)
    sample2_atn_mask = [1] * len_seq2 + [0] * (max_seq_length - len_seq2)

    sample1 = torch.tensor(sample1, dtype=torch.long).unsqueeze(0).cuda()
    sample2 = torch.tensor(sample2, dtype=torch.long).unsqueeze(0).cuda()
    sample1_atn_mask = torch.tensor(sample1_atn_mask, dtype=torch.long).unsqueeze(0).cuda()
    sample2_atn_mask = torch.tensor(sample2_atn_mask, dtype=torch.long).unsqueeze(0).cuda()

    pl_model = get_model()
    model = pl_model.model.eval().to("cpu" if ARGS.cpu else "cuda")
    with torch.no_grad():
        if ARGS.arch != "RPT":
            score1, score2 = model(sample1, sample2)
            prob1 = float(torch.sigmoid(score1.cpu()))
            prob2 = float(torch.sigmoid(score2.cpu()))
            print(prob1, prob2)
        else:
            score1, score2 = model(sample1, sample1_atn_mask, sample2, sample2_atn_mask)
            prob1 = float(torch.sigmoid(score1.cpu()))
            prob2 = float(torch.sigmoid(score2.cpu()))
            print(f"Score for seq1: {prob1:.5f}, Score for seq2: {prob2:.5f}")


def parse_args():
    global ARGS
    ## TRAINING SUBPARSER
    train_subparser = subparsers.add_parser("train", help="Train existing architectures.")
    train_subparser.add_argument(
        "--cpu",
        help="Run on CPU",
        action="store_true",
        default=True,
    )
    train_subparser.add_argument(
        "--rootdir",
        help="Specify the root dir of the training data. Default to '.'",
        type=str,
        default=".",
    )
    train_subparser.add_argument(
        "--arch",
        help="Specify the architecture to train",
        choices=["FC-GPT", "FC-GLOVE", "CNN", "LSTM"],
        type=str,
        required=True,
    )
    train_subparser.add_argument(
        "--feedback",
        help="Specify the feedback to train on",
        choices=["updown", "depth", "width"],
        type=str,
        required=True,
    )
    train_subparser.add_argument(
        "--max-epoch",
        help="Specify the maximum number of epoch to train",
        type=int,
        default=30,
    )
    train_subparser.set_defaults(func=train)

    ## EVALUATING SUBPARSER
    eval_subparser = subparsers.add_parser("eval", help="Evaluate existing architectures.")
    eval_subparser.add_argument(
        "--cpu",
        help="Run on CPU",
        action="store_true",
        default=True,
    )
    eval_subparser.add_argument(
        "--rootdir",
        help="Specify the root dir of the evaluation data. Default to '.'",
        type=str,
        default=".",
    )
    eval_subparser.add_argument(
        "--arch",
        help="Specify the architecture to evaluate",
        choices=["FC-GPT", "FC-GLOVE", "CNN", "LSTM", "RPT"],
        type=str,
        required=True,
    )
    eval_subparser.add_argument(
        "--feedback",
        help="Specify the feedback to evaluate on",
        choices=["updown", "depth", "width"],
        type=str,
        required=True,
    )
    eval_subparser.set_defaults(func=evaluate)

    ## PREDICTING SUBPARSER
    pred_subparser = subparsers.add_parser("predict", help="Predict using existing architectures.")
    pred_subparser.add_argument(
        "--cpu",
        help="Run on CPU",
        action="store_true",
        default=True,
    )
    pred_subparser.add_argument(
        "--rootdir",
        help="Specify the root dir of the project. Default to '.'",
        type=str,
        default=".",
    )
    pred_subparser.add_argument(
        "--arch",
        help="Specify the architecture to do prediction on",
        choices=["FC-GPT", "FC-GLOVE", "CNN", "LSTM", "RPT"],
        type=str,
        required=True,
    )
    pred_subparser.add_argument(
        "--feedback",
        help="Specify the feedback to do prediction on",
        choices=["updown", "depth", "width"],
        type=str,
        required=True,
    )
    pred_subparser.add_argument(
        "--context",
        help="Specify the context",
        type=str,
        required=True,
    )
    pred_subparser.add_argument(
        "--seq1",
        help="Specify sequence 1",
        type=str,
        required=True,
    )
    pred_subparser.add_argument(
        "--seq2",
        help="Specify sequence 2",
        type=str,
        required=True,
    )
    pred_subparser.set_defaults(func=predict)


    ARGS = parser.parse_args()
    ARGS.func()


def main():
    parse_args()


if __name__ == "__main__":
    main()
