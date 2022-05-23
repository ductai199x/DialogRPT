import os
from argparse import ArgumentError, ArgumentParser
from multiprocessing import Manager, Pool

import pandas as pd
import requests
import rich
import tqdm

console = rich.get_console()
parser = ArgumentParser()
MAX_PARALLEL_DOWNLOADS = 6


def load_statistics():
    global stat_df
    stat_df = pd.read_csv(ARGS.stat_file, index_col="DATE")


def get_date_from_range(from_date, to_date):
    from_year, from_month = [int(i) for i in from_date.split("-")]
    to_year, to_month = [int(i) for i in to_date.split("-")]
    if to_year < from_year:
        raise ArgumentError(f"to_year={to_year} < from_year={from_year}")
    elif to_year == from_year:
        if to_month < from_month:
            raise ArgumentError(f"from_date={from_date} < to_date={to_date}")

    dates = []
    m = from_month
    for y in range(from_year, to_year + 1):
        while True:
            dates.append(f"{y}-{m:02d}")
            if (y == to_year) and (m == to_month):
                break
            m += 1
            if m > 12:
                m = 1
                break
    return dates


def query_stats():
    if ARGS.date_list is None:  # using --from-to
        dates = get_date_from_range(*ARGS.from_to)
        rows = stat_df.query(
            f"DATE == {dates} & RC_FILENAME == RC_FILENAME"
        )  # in numpy and pandas, NaN != NaN
        if len(rows) == 0:
            console.print(
                f"[bold red][ERROR]:[/bold red] [yellow]Cannot find any dates "
                + f"from {ARGS.from_to[0]} to {ARGS.from_to[1]} that has both RS and RC [/yellow]",
            )
            exit(1)
        else:
            download(list(rows["RS_FILENAME"]) + list(rows["RC_FILENAME"]))


def downloader(pos, url, fpath, lock):
    response = requests.get(url, stream=True)

    with lock:
        pbar = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=os.path.split(fpath)[1],
            total=int(response.headers.get("content-length", 0)),
            position=pos,
            leave=True,
        )
    with open(fpath, "wb") as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)
            with lock:
                pbar.update(len(chunk))


def download(down_list):
    console.print(f"[bold blue][INFO]: This is the download list:[/bold blue]\n{down_list}")
    download_dir = f"{ARGS.root_dir}/data/compressed"
    download_dir = os.path.abspath(os.path.normpath(download_dir))
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    else:
        if not os.path.isdir(download_dir):
            raise FileExistsError(f"{download_dir} is a file, not a directory!")

    rc_url_prefix = "https://files.pushshift.io/reddit/comments"
    rs_url_prefix = "https://files.pushshift.io/reddit/submissions"
    url_list = []
    for fname in down_list:
        if fname[:2] == "RC":
            url_list.append(f"{rc_url_prefix}/{fname}")
        else:
            url_list.append(f"{rs_url_prefix}/{fname}")
    # Include tqdm bar's position in the down_list
    positions = [i % MAX_PARALLEL_DOWNLOADS for i in list(range(len(down_list)))]
    fpath_list = [f"{download_dir}/{fname}" for fname in down_list]
    downloads_args = list(zip(positions, url_list, fpath_list))

    lock = Manager().Lock()
    with Pool(MAX_PARALLEL_DOWNLOADS) as pool:
        for pos, url, fpath in downloads_args:
            pool.apply_async(downloader, args=(pos, url, fpath, lock))
        pool.close()
        pool.join()


def parse_args():
    global ARGS
    parser.add_argument(
        "-s",
        "--stat-file",
        help="the path to the statistics.csv file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        help="the path to the root dir of the project. "
        + "Files will be downloaded to <root_dir>/data/compressed",
        type=str,
        default=".",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f",
        "--from-to",
        help="the year and month to download FROM and TO. Format: %%Y-%%m",
        type=str,
        nargs=2,
    )
    group.add_argument(
        "-d",
        "--date-list",
        help="List of dates (as %%Y-%%m) to download.",
        type=str,
        nargs="+",
    )
    ARGS = parser.parse_args()


def main():
    parse_args()
    load_statistics()
    query_stats()


if __name__ == "__main__":
    main()
