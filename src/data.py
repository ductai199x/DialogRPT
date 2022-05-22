import bz2
import json
import lzma
import os
import pickle
import time
from argparse import ArgumentParser
from multiprocessing import Manager, Pool

import numpy as np
import rich
import zstandard
from blessings import Terminal
from tqdm.auto import tqdm

parser = ArgumentParser()
console = rich.get_console()
term = Terminal()
LINE_UP = "\033[1A"
LINE_CLEAR = "\033[K"
MAX_PARALLEL_PROCS = 8
TOP_K_TEXTS = 30


def print_mult_procs(msg, lock, pos):
    with lock:
        with term.location(0, term.height - pos - 2):
            print(end=LINE_CLEAR, flush=True)
            print(msg, flush=True)


def dump_queue(q):
    q.put(None)
    return list(iter(lambda: q.get(timeout=0.00001), None))


def get_all_files(path, prefix="", suffix="", contains=("",), excludes=("",)):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    files = []
    for pre, dirs, basenames in os.walk(path):
        for name in basenames:
            if name.startswith(prefix) and name.endswith(suffix) and any([c in name for c in contains]):
                if excludes == ("",):
                    files.append(os.path.join(pre, name))
                else:
                    if all([e not in name for e in excludes]):
                        files.append(os.path.join(pre, name))
    return files


def extract_zst(archive: str, out_path: str, pos: int, overwrite: bool, lock: Manager()):
    """extract .zst file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: str
        .zst file to extract
    out_path: str
        directory to extract files and directories to
    pos: int
        the position of the tqdm progress bar on the terminal
    overwrite: bool
        flag to overwrite the file if exists
    lock: SyncManager
        a lock to print the tqdm bar to the terminal asynchronously
    """
    with lock:
        if os.path.exists(out_path) and out_path != os.devnull:
            if os.path.isfile(out_path):
                if not overwrite:
                    tqdm.write(f"[WARN]: {out_path} exists. Skipping.")
                    return
            else:
                tqdm.write(f"[ERROR]: {out_path} exists but it's not a file!")
                raise FileExistsError

    with lock:
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"Extracting {os.path.split(archive)[1]}",
            position=pos,
            leave=False,
        )

    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
    with open(archive, "rb") as input_file, open(out_path, "wb") as out_file:
        for chunk in dctx.read_to_iter(input_file, read_size=2000 * 1024):
            out_file.write(chunk)
            with lock:
                pbar.update(len(chunk))

    with lock:
        pbar.close()


def extract_bz2(archive: str, out_path: str, pos: int, overwrite: bool, lock: Manager()):
    """extract .bz2 file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: str
        .bz2 file to extract
    out_path: str
        directory to extract files and directories to
    pos: int
        the position of the tqdm progress bar on the terminal
    overwrite: bool
        flag to overwrite the file if exists
    lock: SyncManager
        a lock to print the tqdm bar to the terminal asynchronously
    """
    with lock:
        if os.path.exists(out_path) and out_path != os.devnull:
            if os.path.isfile(out_path):
                if not overwrite:
                    tqdm.write(f"[WARN]: {out_path} exists. Skipping.")
                    return
            else:
                tqdm.write(f"[ERROR]: {out_path} exists but it's not a file!")
                raise FileExistsError

    with lock:
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"Extracting {os.path.split(archive)[1]}",
            position=pos,
            leave=False,
        )

    with bz2.BZ2File(archive, "rb") as input_file, open(out_path, "wb") as out_file:
        for chunk in iter(lambda: input_file.read(2000 * 1024), b""):
            out_file.write(chunk)
            with lock:
                pbar.update(len(chunk))

    with lock:
        pbar.close()


def extract_xz(archive: str, out_path: str, pos: int, overwrite: bool, lock: Manager()):
    """extract .xz file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: str
        .xz file to extract
    out_path: str
        directory to extract files and directories to
    pos: int
        the position of the tqdm progress bar on the terminal
    overwrite: bool
        flag to overwrite the file if exists
    lock: SyncManager
        a lock to print the tqdm bar to the terminal asynchronously
    """
    with lock:
        if os.path.exists(out_path) and out_path != os.devnull:
            if os.path.isfile(out_path):
                if not overwrite:
                    tqdm.write(f"[WARN]: {out_path} exists. Skipping.")
                    return
            else:
                tqdm.write(f"[ERROR]: {out_path} exists but it's not a file!")
                raise FileExistsError

    with lock:
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"Extracting {os.path.split(archive)[1]}",
            position=pos,
            leave=False,
        )

    with lzma.LZMAFile(archive, "rb") as input_file, open(out_path, "wb") as out_file:
        for chunk in iter(lambda: input_file.read(2000 * 1024), b""):
            out_file.write(chunk)
            with lock:
                pbar.update(len(chunk))

    with lock:
        pbar.close()


def valid_sub(sub):
    if sub.upper() in [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]:
        # not allowed by Windows system
        return False
    if ":" in sub:
        return False
    return True


def get_dates(year):
    # search for the year in the compressed files
    compressed_files = get_all_files(compressed_dir)
    dates = [os.path.splitext(os.path.split(path)[1])[0][3:] for path in compressed_files]
    dates = list(set([d for d in dates if str(year) in d]))
    if len(dates) == 0:
        raise RuntimeError(
            f"No {year} available in {compressed_dir}. Please download both RS's and RC's for that year."
        )
    return dates


def get_extract_method(ext):
    if ext == ".bz2":
        return extract_bz2
    elif ext == ".zst":
        return extract_zst
    if ext == ".xz":
        return extract_xz


def extract_rc(date, print_pos, lock):
    extracted_path = f"{compressed_dir}/RC_{date}.extracted"

    nodes = dict()
    edges = dict()
    subs = set()
    n = 0
    m = 0
    kk = ["body", "link_id", "name", "parent_id", "subreddit"]

    def save(nodes, edges):
        for sub in nodes:
            dir = f"{jsonl_dir}/{sub}"
            try:
                os.makedirs(dir, exist_ok=True)
            except NotADirectoryError as e:
                print(e)
                continue
            if sub not in subs:
                with open(f"{dir}/{date}_nodes.jsonl", "w", encoding="utf-8") as f:
                    pass
                with open(f"{dir}/{date}_edges.tsv", "w", encoding="utf-8") as f:
                    pass
                subs.add(sub)
            with open(f"{dir}/{date}_nodes.jsonl", "a", encoding="utf-8") as f:
                f.write("\n".join(nodes[sub]) + "\n")
            with open(f"{dir}/{date}_edges.tsv", "a", encoding="utf-8") as f:
                f.write("\n".join(edges[sub]) + "\n")

    with open(extracted_path, "r", encoding="utf-8") as extracted_file:
        with lock:
            pbar = tqdm(
                total=sum(1 for l in open(extracted_path)),
                position=print_pos,
                ncols=120,
                leave=False,
            )
        for line in extracted_file:
            n += 1
            line = line.strip("\n")
            try:
                node = json.loads(line)
            except Exception:
                continue

            ok = True
            for k in kk:
                if k not in node:
                    ok = False
                    break
            if not ok:
                break

            if not valid_sub(node["subreddit"]):
                continue

            if node["subreddit"] not in nodes:
                nodes[node["subreddit"]] = []
                edges[node["subreddit"]] = []
            nodes[node["subreddit"]].append(line)
            edges[node["subreddit"]].append(f"{node['link_id']}\t{node['parent_id']}\t{node['name']}")

            m += 1
            if m % 2e4 == 0:
                save(nodes, edges)
                nodes = dict()
                edges = dict()
                with lock:
                    pbar.set_postfix_str(
                        f"[RC_{date}] saved {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits"
                    )
                    pbar.update(2e4)

    save(nodes, edges)
    with lock:
        pbar.clear()
        pbar.close()
        pbar.display(f"[RC_{date}] FINAL {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits ================")

    with open(f"{jsonl_dir}/readme.txt", "a", encoding="utf-8") as f:
        f.write(f"[{date}] saved {m}/{n}\n")


def extract_rs(date, print_pos, lock):
    extracted_path = f"{compressed_dir}/RS_{date}.extracted"

    roots = dict()
    subs = set()
    n = 0
    m = 0
    kk = ["selftext", "id", "title", "subreddit"]

    def save(roots):
        for sub in roots:
            dir = f"{jsonl_dir}/{sub}"
            try:
                os.makedirs(dir, exist_ok=True)
            except NotADirectoryError as e:
                print(e)
                continue
            if sub not in subs:
                with open(f"{dir}/{date}_roots.jsonl", "w", encoding="utf-8") as f:
                    pass
                subs.add(sub)
            with open(f"{dir}/{date}_roots.jsonl", "a", encoding="utf-8") as f:
                f.write("\n".join(roots[sub]) + "\n")

    with open(extracted_path, "r", encoding="utf-8") as extracted_file:
        with lock:
            pbar = tqdm(
                total=sum(1 for l in open(extracted_path)),
                position=print_pos,
                ncols=120,
                leave=False,
            )
        for line in extracted_file:
            n += 1
            line = line.strip("\n")
            try:
                root = json.loads(line)
            except Exception:
                continue

            ok = True
            for k in kk:
                if k not in root:
                    ok = False
                    break
            if not ok:
                break
            if not valid_sub(root["subreddit"]):
                continue

            # some bz2, e.g. 2012-09, doesn"t have the `name` entry
            if "name" not in root:
                root["name"] = f"t3_{root['id']}"

            if root["subreddit"] not in roots:
                roots[root["subreddit"]] = []
            roots[root["subreddit"]].append(line)

            m += 1
            if m % 2e4 == 0:
                save(roots)
                roots = dict()
                with lock:
                    pbar.set_postfix_str(
                        f"[RS_{date}] saved {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits"
                    )
                    pbar.update(2e4)
    save(roots)
    with lock:
        pbar.clear()
        pbar.close()
        pbar.display(f"[RS_{date}] FINAL {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits ================")

    with open(f"{jsonl_dir}/readme_roots.txt", "a", encoding="utf-8") as f:
        f.write(f"[{date}] saved {m}/{n}\n")


def extract_txt(sub, year, pos_queue, lock, tokenizer, result_queue, overwrite=True, max_subword=3):
    pos = pos_queue.get()
    print = lambda msg: print_mult_procs(msg, lock, pos)
    print(f"[{sub:<30} {year:<7}] Extracting Texts")

    dir = f"{redditsub_dir}/{sub}"
    os.makedirs(dir, exist_ok=True)
    path_out = f"{dir}/{year}_txt.tsv"
    path_done = f"{path_out}.done"
    if not overwrite and os.path.exists(path_done):
        return

    dates = get_dates(year)
    open(path_out, "w", encoding="utf-8")

    def clean(txt):
        if txt.strip() in ["[deleted]", "[removed]"]:
            return None
        if ">" in txt or "&gt;" in txt:  # no comment in line ("&gt;" means ">")
            return None

        # deal with URL
        txt = txt.replace("](", "] (")
        ww = []
        for w in txt.split():
            if len(w) == 0:
                continue
            if "://" in w.lower() or "http" in w.lower():
                ww.append("(URL)")
            else:
                ww.append(w)
        if not ww:
            return None
        if len(ww) > 30:  # focus on dialog, so ignore long txt
            return None
        if len(ww) < 1:
            return None
        txt = " ".join(ww)
        for c in ["\t", "\n", "\r"]:  # delimiter or newline
            txt = txt.replace(c, " ")
        ids = tokenizer.encode(txt, max_length=1024)
        if len(ids) / len(ww) > max_subword:  # usually < 1.5. too large means too many unknown words
            return None

        ids = " ".join([str(x) for x in ids])
        return txt, ids

    lines = []
    m = 0
    n = 0
    name_set = set()
    for date in dates:
        path = f"{jsonl_dir}/{sub}/{date}_nodes.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path, "r", encoding="utf-8"):
            n += 1
            d = json.loads(line.strip("\n"))
            if d["name"] in name_set:
                continue
            name_set.add(d["name"])
            txt_ids = clean(d["body"])
            if txt_ids is not None:
                txt, ids = txt_ids
                lines.append(f"{d['name']}\t{txt}\t{ids}")
                m += 1

            if m > 0 and m % 1e4 == 0:
                print(f"[{sub:<30} {date}] Number of lines accepted {m}/{n}")
                with open(path_out, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                lines = []

    for date in dates:
        path = f"{jsonl_dir}/{sub}/{date}_roots.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path, "r", encoding="utf-8"):
            n += 1
            d = json.loads(line.strip("\n"))
            if "name" not in d:
                d["name"] = f"t3_{d['id']}"
            if d["name"] in name_set:
                continue
            name_set.add(d["name"])
            txt_ids = clean(f"{d['title']} {d['selftext']}")
            if txt_ids is not None:
                txt, ids = txt_ids
                lines.append(f"{d['name']}\t{txt}\t{ids}")
                m += 1
            if m % 1e4 == 0:
                print(f"[{sub:<30} {date}] Number of lines accepted {m}/{n}")
                with open(path_out, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                lines = []
    if lines:
        with open(path_out, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    s = f"[{sub:<30} {year:<7}] Number of lines accepted {m}/{n}"
    with open(path_done, "w") as f:
        f.write(s)
        print(s)
        result_queue.append((sub, m))
        pos_queue.put(pos)


def extract_trees(sub, year, pos_queue, lock, overwrite=True):
    pos = pos_queue.get()
    print = lambda msg: print_mult_procs(msg, lock, pos)
    print(f"[{sub}-{year}] Extracting Trees")

    dir = f"{redditsub_dir}/{sub}"
    os.makedirs(dir, exist_ok=True)
    path_out = f"{dir}/{year}_trees.pkl"
    if os.path.exists(path_out) and not overwrite:
        return

    trees = dict()
    n = 0
    for date in get_dates(year):
        path = f"{jsonl_dir}/{sub}/{date}_edges.tsv"
        if not os.path.exists(path):
            # print("no such file: "+path)
            continue
        for line in open(path, "r", encoding="utf-8"):
            n += 1
            link, parent, child = line.strip("\n").split("\t")
            if link not in trees:
                trees[link] = dict()
            trees[link][(parent, child)] = date

    if not trees:
        return

    os.makedirs(dir, exist_ok=True)
    pickle.dump(trees, open(path_out, "wb"))
    print(f"[{sub}-{year}] {len(trees)} trees {n/len(trees):.1f} nodes/tree")
    pos_queue.put(pos)


def extract_time(sub, year, pos_queue, lock, overwrite=True):
    pos = pos_queue.get()
    print = lambda msg: print_mult_procs(msg, lock, pos)
    print(f"[{sub}-{year}] Extracting Time")

    dir = f"{redditsub_dir}/{sub}"
    os.makedirs(dir, exist_ok=True)
    path_out = f"{dir}/{year}_time.tsv"
    path_done = f"{path_out}.done"
    if not overwrite and os.path.exists(path_done):
        return
    dates = get_dates(year)
    os.makedirs(dir, exist_ok=True)
    open(path_out, "w", encoding="utf-8")

    lines = []
    m = 0
    n = 0
    name_set = set()
    for date in dates:
        path = f"{jsonl_dir}/{sub}/{date}_nodes.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path, "r", encoding="utf-8"):
            n += 1
            d = json.loads(line.strip("\n"))
            if "name" not in d:
                d["name"] = f"t3_{d['id']}"
            if d["name"] in name_set:
                continue
            name_set.add(d["name"])
            t = d["created_utc"]
            lines.append(f"{d['name']}\t{t}")
            m += 1
            if m % 1e4 == 0:
                with open(path_out, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                lines = []
    with open(path_out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    s = f"[{sub}-{year}] time kept {m}/{n}"
    with open(path_done, "w") as f:
        f.write(s)
        print(s)
        pos_queue.put(pos)


def extract_feedback(sub, year, pos_queue, lock, overwrite=True):
    pos = pos_queue.get()
    print = lambda msg: print_mult_procs(msg, lock, pos)
    print(f"[{sub}-{year}] Extracting Feedback")

    dir = f"{redditsub_dir}/{sub}"
    path_out = f"{dir}/{year}_feedback.tsv"
    path_done = f"{path_out}.done"
    if not overwrite and os.path.exists(path_done):
        return

    path_pkl = f"{dir}/{year}_trees.pkl"
    if not os.path.exists(path_pkl):
        return
    trees = pickle.load(open(path_pkl, "rb"))
    if not trees:
        return

    dates = get_dates(year)
    updown = dict()
    for date in dates:
        path = f"{jsonl_dir}/{sub}/{date}_nodes.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path, "r", encoding="utf-8"):
            d = json.loads(line.strip("\n"))
            updown[d["name"]] = d["ups"] - d["downs"]

    if not updown:
        print("empty updown")
        return

    with open(path_out, "w", encoding="utf-8") as f:
        f.write("\t".join(["#path", "vol", "width", "depth", "updown"]) + "\n")

    print(f"[{sub}-{year}] calculating scores for {len(trees)} trees...")

    n_tree = 0
    n_node = 0
    for root in trees:
        tree = trees[root]
        children = dict()
        for parent, child in tree:
            if parent not in children:
                children[parent] = []
            children[parent].append(child)
        if root not in children:
            continue

        # BFS to get all paths from root to leaf
        q = [[root]]
        paths = []
        while q:
            qsize = len(q)
            for _ in range(qsize):
                path = q.pop(0)
                head = path[-1]
                if head not in children:  # then head is a leaf
                    paths.append(path)
                    continue
                for child in children[head]:
                    q.append(path + [child])

        prev = dict()
        for path in paths:
            for i in range(1, len(path)):
                prev[path[i]] = " ".join(path[: i + 1])

        descendant = dict()
        longest_subpath = dict()
        while paths:
            path = paths.pop(0)
            node = path[0]
            if node not in descendant:
                descendant[node] = set()
                longest_subpath[node] = 0
            descendant[node] |= set(path[1:])
            longest_subpath[node] = max(longest_subpath[node], len(path) - 1)
            if len(path) > 1:
                paths.append(path[1:])

        sorted_nodes = sorted([(len(prev[node].split()), prev[node], node) for node in prev])
        if not sorted_nodes:
            continue

        n_tree += 1
        lines = []
        for _, _, node in sorted_nodes:
            if node == root:
                continue
            if node not in updown:
                continue
            n_node += 1
            lines.append(
                "%s\t%i\t%i\t%i\t%i"
                % (
                    prev[node],  # turns:    path from its root to this node
                    len(descendant[node]),  # vol:      num of descendants of this node
                    len(children.get(node, [])),  # width:    num of direct childrent of this node
                    longest_subpath[node],  # depth:    num of longest subpath of this node
                    updown[node],  # updown:   `upvotes - downvotes` of this node
                )
            )
        with open(path_out, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    if n_tree:
        s = f"[{sub}-{year}] {n_tree} tree {n_node} nodes"
    else:
        s = f"[{sub}-{year}] trees are empty!"
    with open(path_done, "w") as f:
        f.write(s)
        print(s)
        pos_queue.put(pos)


def create_pairs(year, sub, feedback, overwrite=True):
    dir = f"{redditsub_dir}/{sub}"
    path_out = f"{dir}/{year}_{feedback}.tsv"
    path_done = f"{path_out}.done"
    if not overwrite and os.path.exists(path_done):
        return

    ix_feedback = ["vol", "width", "depth", "updown"].index(feedback) + 1
    path_in = f"{dir}/{year}_feedback.tsv"
    if not os.path.exists(path_in):
        return

    time = dict()
    path_time = f"{dir}/{year}_time.tsv"
    if not os.path.exists(path_time):
        return
    for line in open(path_time, "r"):
        ss = line.strip("\n").split("\t")
        if len(ss) == 2:
            name, t = ss
            time[name] = int(t)

    open(path_out, "w", encoding="utf-8")
    print(f"[{sub}-{year}] creating pairs...")

    def match_time(replies, cxt):
        scores = sorted(set([score for score, _ in replies]))
        m = len(scores)
        if m < 2:
            return 0  # can"t create pairs if m < 2
        cand = []
        for score, reply in replies:
            if reply not in time:
                continue
            cand.append((time[reply], score, reply))
        cand = sorted(cand)
        rank = [scores.index(score) / (m - 1) for _, score, _ in cand]
        lines = []
        for i in range(len(cand) - 1):
            t_a, score_a, a = cand[i]
            t_b, score_b, b = cand[i + 1]
            rank_a = rank[i]
            rank_b = rank[i + 1]
            if score_a == score_b:
                continue
            hr = (t_b - t_a) / 3600
            if score_b > score_a:
                score_a, score_b = score_b, score_a
                a, b = b, a
                rank_a, rank_b = rank_b, rank_a
            lines.append(
                "\t".join(
                    [
                        cxt,
                        a,
                        b,
                        "%.2f" % hr,
                        "%i" % score_a,
                        "%i" % score_b,
                        "%.4f" % rank_a,
                        "%.4f" % rank_b,
                    ]
                )
            )
        if lines:
            with open(path_out, "a") as f:
                f.write("\n".join(lines) + "\n")
        return len(lines)

    n_line = 0
    prev = None
    replies = []
    for line in open(path_in, "r"):
        if line.startswith("#"):
            continue
        ss = line.strip("\n").split("\t")
        turns = ss[0].split()  # including both cxt and resp
        if len(turns) < 2:
            continue
        reply = turns[-1]
        try:
            score = int(ss[ix_feedback])
        except ValueError:
            continue
        parent = turns[-2]
        if parent == prev:
            replies.append((score, reply))
        else:
            if replies:
                n_line += match_time(replies, cxt)
            cxt = " ".join(turns[:-1])
            prev = parent
            replies = [(score, reply)]
    if replies:
        n_line += match_time(replies, cxt)

    s = f"[{sub}-{year} {feedback}] {n_line} pairs"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def add_seq(sub, year, feedback, overwrite=False):
    fname = f"{year}_{feedback}"
    dir = f"{redditsub_dir}/{sub}"
    turn_sep = " 50256 "
    path_out = f"{dir}/{fname}_ids.tsv"
    path_done = f"{path_out}.done"

    if os.path.exists(path_done) and not overwrite:
        return
    if not os.path.exists(f"{dir}/{fname}.tsv"):
        return

    seq = dict()
    path = f"{dir}/{year}_txt.tsv"
    if not os.path.exists(path):
        return
    for line in open(path, "r", encoding="utf-8"):
        ss = line.strip("\n").split("\t")
        if len(ss) != 3:
            continue
        name, txt, ids = ss
        seq[name] = ids

    print(f"Loaded {len(seq)} seq")
    with open(path_out, "w", encoding="utf-8") as f:
        pass
    print(f"[{sub}-{year} {feedback}] adding seq")

    lines = []
    n = 0
    m = 0
    path = f"{dir}/{fname}.tsv"
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip("\n")
        if line.startswith("#"):
            continue

        n += 1
        ss = line.split("\t")
        if len(ss) < 7:
            continue
        name_cxt, name_pos, name_neg = ss[:3]

        cxt = []
        ok = True
        for name in name_cxt.split():
            if name in seq:
                cxt.append(seq[name])
            else:
                ok = False
                break
        if not ok:
            continue
        cxt = turn_sep.join(cxt)

        if name_pos in seq:
            reply_pos = seq[name_pos]
        else:
            continue
        if name_neg in seq:
            reply_neg = seq[name_neg]
        else:
            continue

        lines.append(
            "\t".join(
                [
                    cxt,
                    reply_pos,
                    reply_neg,
                    name_cxt,
                    name_pos,
                    name_neg,
                ]
                + ss[3:]
            )
        )
        m += 1
        if m % 1e4 == 0:
            with open(path_out, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            lines = []

    with open(path_out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    s = f"[{sub}-{year} {feedback}] pair seq {m}/{n}"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def combine_sub(year, feedback, overwrite=False, skip_same_pos=True):
    dir = f"{output_dir}/{feedback}/{year}"
    os.makedirs(dir, exist_ok=True)
    path_out = f"{dir}/raw.tsv"
    path_done = f"{path_out}.done"
    if os.path.exists(path_done) and not overwrite:
        return path_out

    subs = sorted(os.listdir(redditsub_dir))
    open(path_out, "w", encoding="utf-8")
    lines = []
    n = 0
    empty = True
    non_empty_subreddits = 0
    for sub in subs:
        empty = True
        path = f"{redditsub_dir}/{sub}/{year}_{feedback}_ids.tsv"
        if not os.path.exists(path):
            continue
        for line in open(path, "r", encoding="utf-8"):
            if line.startswith("#"):
                continue
            line = line.strip("\n")
            if not line:
                continue
            lines.append(line)
            empty = False
            n += 1
            if n % 1e5 == 0:
                with open(path_out, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                lines = []
                print(
                    f"[{year} {feedback}] saved {n/1e6:.2f} M lines from {non_empty_subreddits+1} subreddits, now is {sub}"
                )
        if not empty:
            non_empty_subreddits += 1

    with open(path_out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    s = f"[{year} {feedback}] saved {n/1e6:.2f} M lines from {non_empty_subreddits} subreddits"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)
    return path_out


def split_by_root(path, p_test=0.15):
    print(f"Spliting by root path: {path}, Test vs. Val ratio: {1-p_test}-{p_test}")
    datasets = {
        "train": [],
        "vali": [],
    }
    prev = None
    n = 0

    for set_ in datasets:
        with open(f"{path}.{set_}", "w", encoding="utf-8") as f:
            pass

    for line in open(path, "r", encoding="utf-8"):
        line = line.strip("\n")
        if not line:
            continue
        cxt = line.split("\t")[3]
        root = cxt.strip().split()[0]
        if root != prev:
            if np.random.random() < p_test:
                set_ = "vali"
            else:
                set_ = "train"
        datasets[set_].append(line)
        prev = root
        n += 1
        if n % 1e6 == 0:
            for set_ in datasets:
                if len(datasets[set_]) == 0:
                    continue
                with open(f"{path}.{set_}", "a", encoding="utf-8") as f:
                    f.write("\n".join(datasets[set_]) + "\n")
                datasets[set_] = []

    print(f"Test vs. Val samples: {len(datasets['train'])}-{len(datasets['vali'])}")
    for set_ in datasets:
        if len(datasets[set_]) == 0:
            continue
        with open(f"{path}.{set_}", "a", encoding="utf-8") as f:
            f.write("\n".join(datasets[set_]))


def shuffle(year, feedback, part, n_temp=10):
    dir = f"{output_dir}/{feedback}/{year}"
    path = f"{dir}/raw.tsv.{part}"
    path_out = f"{dir}/{part}.tsv"
    dir_temp = f"{output_dir}/temp/{feedback}/{year}"

    print(f"Shuffling {path}...")
    os.makedirs(dir_temp, exist_ok=True)
    lines = [[] for _ in range(n_temp)]

    # split into n_temp files
    for i in range(n_temp):
        with open(f"{dir_temp}/temp{i}", "w", encoding="utf-8") as f:
            pass

    n = 0
    count = [0] * n_temp
    rand = np.random.randint(0, n_temp, 202005)
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip("\n")
        if len(line) == 0:
            continue
        bucket = rand[n % len(rand)]
        lines[bucket].append(line)
        count[bucket] += 1
        n += 1
        if n % 1e6 == 0:
            print(f"read {n/1e6:.2f} M")
            for i in range(n_temp):
                if len(lines[i]) == 0:
                    continue
                with open(f"{dir_temp}/temp{i}", "a", encoding="utf-8") as f:
                    f.write("\n".join(lines[i]) + "\n")
                lines[i] = []

    # write the remaining lines
    for i in range(n_temp):
        with open(f"{dir_temp}/temp{i}", "a", encoding="utf-8") as f:
            f.write("\n".join(lines[i]))

    # and then merge
    open(path_out, "w", encoding="utf-8")
    for i in range(n_temp):
        lines = open(f"{dir_temp}/temp{i}", "r", encoding="utf-8").readlines()
        jj = list(range(len(lines)))
        np.random.shuffle(jj)
        with open(path_out, "a", encoding="utf-8") as f:
            f.write("\n".join([lines[j].strip("\n") for j in jj]) + "\n")
        os.remove(f"{dir_temp}/temp{i}")

    print(f"Done with {path}. Cleaning up.")
    os.removedirs(f"{dir_temp}")


def get_subs():
    # return ["4chan"]
    tqdm.write("Collectiing subs...")
    subs = sorted(os.listdir(jsonl_dir))
    tqdm.write(f"Collected {len(subs)} subs")
    return subs


def build_json(year, overwrite=True):
    dates = get_dates(year)
    if len(dates) == 0:
        console.print(
            "[ERROR]: [yellow]No dates for that year is available. "
            + "Please check your data/compressed directory"
        )
        return False

    # Find and Extract all the zip files for that year
    RS_files = get_all_files(f"{compressed_dir}", prefix="RS", contains=(str(year),), excludes=("extracted",))
    RC_files = get_all_files(f"{compressed_dir}", prefix="RC", contains=(str(year),), excludes=("extracted",))
    extract_list = RS_files + RC_files
    extract_pos = [i % MAX_PARALLEL_PROCS for i in list(range(len(extract_list)))]
    extract_to = [f"{os.path.splitext(fpath)[0]}.extracted" for fpath in extract_list]
    extract_args = list(zip(extract_pos, extract_list, extract_to))

    lock = Manager().Lock()
    with Pool(MAX_PARALLEL_PROCS) as pool:
        for pos, fpath, fpath_to in extract_args:
            _, ext = os.path.splitext(fpath)
            extract_fn = get_extract_method(ext)
            pool.apply_async(extract_fn, args=(fpath, fpath_to, pos, overwrite, lock))
        pool.close()
        pool.join()

    with Pool(MAX_PARALLEL_PROCS) as pool:
        for idx, date in enumerate(dates):
            pos = idx % MAX_PARALLEL_PROCS
            pool.apply_async(extract_rc, args=(date, pos, lock))
        pool.close()
        pool.join()

    with Pool(MAX_PARALLEL_PROCS) as pool:
        for idx, date in enumerate(dates):
            pos = idx % MAX_PARALLEL_PROCS
            pool.apply_async(extract_rs, args=(date, pos, lock))
        pool.close()
        pool.join()

    return True


def build_basic(year, overwrite=True):
    top_k_subs_fpath = f"{redditsub_dir}/top_k_list.csv"
    if os.path.exists(top_k_subs_fpath) and os.path.isfile(top_k_subs_fpath) and not overwrite:
        with open(top_k_subs_fpath, "r") as f:
            lines = f.readlines()
            if len(lines) == TOP_K_TEXTS:
                top_k_subs = [line.split(",")[0] for line in lines]
                return top_k_subs

    from transformers19 import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, max_length=1024, truncation=True)
    subs = get_subs()
    lock = Manager().Lock()
    result_queue = Manager().list()
    print_pos_queue = Manager().Queue(maxsize=MAX_PARALLEL_PROCS)
    [print_pos_queue.put(i) for i in range(MAX_PARALLEL_PROCS)]

    print("\n" * (MAX_PARALLEL_PROCS), flush=True)

    with term.location(0, term.height - MAX_PARALLEL_PROCS - 2):
        print(end=LINE_CLEAR, flush=True)
        print(f"Extracting Texts (Truncate to top only {TOP_K_TEXTS})...")
    with Pool(MAX_PARALLEL_PROCS) as pool:
        for sub in subs:
            pool.apply_async(
                extract_txt, args=(sub, year, print_pos_queue, lock, tokenizer, result_queue, overwrite)
            )
        pool.close()
        pool.join()

    subs_txt_result_queue = list(result_queue)
    subs_txt_result_queue = sorted(subs_txt_result_queue, reverse=True, key=lambda x: x[1])[:TOP_K_TEXTS]

    with open(top_k_subs_fpath, "w") as f:
        f.write("\n".join([",".join(map(str, i)) for i in subs_txt_result_queue]))

    top_k_subs = [s for s, _ in subs_txt_result_queue]

    with term.location(0, term.height - MAX_PARALLEL_PROCS - 2):
        print(end=LINE_CLEAR, flush=True)
        print("Extracting Time...")
    with Pool(MAX_PARALLEL_PROCS) as pool:
        for sub in top_k_subs:
            pool.apply_async(extract_time, args=(sub, year, print_pos_queue, lock, overwrite))
        pool.close()
        pool.join()

    with term.location(0, term.height - MAX_PARALLEL_PROCS - 2):
        print(end=LINE_CLEAR, flush=True)
        print("Extracting Trees...")
    with Pool(MAX_PARALLEL_PROCS) as pool:
        for sub in top_k_subs:
            pool.apply_async(extract_trees, args=(sub, year, print_pos_queue, lock, overwrite))
        pool.close()
        pool.join()

    with term.location(0, term.height - MAX_PARALLEL_PROCS - 2):
        print(end=LINE_CLEAR, flush=True)
        print("Extracting Feedbacks...")
    with Pool(MAX_PARALLEL_PROCS) as pool:
        for sub in top_k_subs:
            pool.apply_async(extract_feedback, args=(sub, year, print_pos_queue, lock, overwrite))
        pool.close()
        pool.join()

    return top_k_subs


def build_pairs(year, subs, feedback, overwrite):
    for sub in subs:
        create_pairs(year, sub, feedback, overwrite)
        add_seq(sub, year, feedback, overwrite)
    path = combine_sub(year, feedback, overwrite)
    split_by_root(path)
    for part in ["train", "vali"]:
        shuffle(year, feedback, part)


def data_preprocess():
    global root_dir, data_dir, compressed_dir, jsonl_dir, redditsub_dir, output_dir
    root_dir = os.path.abspath(os.path.normpath(ARGS.root_dir))
    data_dir = f"{root_dir}/data"
    compressed_dir = f"{data_dir}/compressed"
    jsonl_dir = f"{data_dir}/json"
    redditsub_dir = f"{data_dir}/subs"
    output_dir = f"{data_dir}/out"

    years = []
    for y in ARGS.year:
        if "-" in y:
            from_y, to_y = y.split("-")
            years += list(range(from_y, to_y + 1))
        else:
            years += [y]

    for year in years:
        # build_status = build_json(year, overwrite=False)
        top_k_subs = build_basic(year, overwrite=True)
        print(f"Building pairs for these {TOP_K_TEXTS} subs: {top_k_subs}")
        # [build_pairs(year, top_k_subs, fb, overwrite=True) for fb in ("updown", "depth", "width")]


def parse_args():
    global ARGS
    parser.add_argument(
        "-r",
        "--root-dir",
        help="the root directory of the project. Default: '.'",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-y",
        "--year",
        help="the year to be processed. Enter one or multiple years separated by space, or a range %Y-%Y",
        type=str,
        nargs="+",
        required=True,
    )
    ARGS = parser.parse_args()


def main():
    parse_args()
    data_preprocess()


if __name__ == "__main__":
    main()
