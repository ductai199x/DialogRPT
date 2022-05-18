# author: Xiang Gao at Microsoft Research AI NLP Group


import bz2
import json
import lzma
import os
import pickle

import numpy as np
import zstandard
from tqdm.auto import tqdm

root_dir = "."
data_dir = f"{root_dir}/data"
compressed_dir = f"{data_dir}/compressed"
jsonl_dir = f"{data_dir}/json"
redditsub_dir = f"{data_dir}/subs"
output_dir = f"{data_dir}/out"


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


def extract_zst(archive: str, out_path: str):
    """extract .zst file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: str
      .zst file to extract
    out_path: str
      directory to extract files and directories to
    """

    if zstandard is None:
        raise ImportError("pip install zstandard")

    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)

    with open(archive, "rb") as input_file, open(out_path, "wb") as out_file:
        for chunk in tqdm(
            dctx.read_to_iter(input_file, read_size=1000 * 1024), desc=f"Extracting {archive}: "
        ):
            out_file.write(chunk)


def extract_bz2(archive: str, out_path: str):
    """extract .bz2 file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: str
        .bz2 file to extract
    out_path: str
        directory to extract files and directories to
    """

    with bz2.BZ2File(archive, "rb") as input_file, open(out_path, "wb") as out_file:
        for data in tqdm(iter(lambda: input_file.read(1000 * 1024), b""), desc=f"Extracting {archive}: "):
            out_file.write(data)


def extract_xz(archive: str, out_path: str):
    """extract .xz file
    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: str
        .xz file to extract
    out_path: str
        directory to extract files and directories to
    """

    with lzma.LZMAFile(archive, "rb") as input_file, open(out_path, "wb") as out_file:
        for data in tqdm(iter(lambda: input_file.read(1000 * 1024), b""), desc=f"Extracting {archive}: "):
            out_file.write(data)


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


def extract_rc(date, is_extracted=False):
    fpath = get_all_files(f"{compressed_dir}", prefix="RC", contains=(date,), excludes=("extracted",))[0]
    fdir, fname = os.path.split(fpath)
    fbasename, fext = os.path.splitext(fname)
    extract_fn = get_extract_method(fext)
    extracted_path = f"{compressed_dir}/{fbasename}.extracted"

    if not is_extracted: extract_fn(fpath, extracted_path)

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
            if m % 1e5 == 0:
                save(nodes, edges)
                print(f"[RC_{date}] saved {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits")
                nodes = dict()
                edges = dict()

    save(nodes, edges)
    print(f"[RC_{date}] FINAL {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits ================")
    with open(f"{jsonl_dir}/readme.txt", "a", encoding="utf-8") as f:
        f.write(f"[{date}] saved {m}/{n}\n")


def extract_rs(date, is_extracted=False):
    fpath = get_all_files(f"{compressed_dir}", prefix="RS", contains=(date,), excludes=("extracted",))[0]
    fdir, fname = os.path.split(fpath)
    fbasename, fext = os.path.splitext(fname)
    extract_fn = get_extract_method(fext)
    extracted_path = f"{compressed_dir}/{fbasename}.extracted"

    if not is_extracted: extract_fn(fpath, extracted_path)

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
            if m % 1e4 == 0:
                save(roots)
                print(f"[RS_{date}] saved {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits")
                roots = dict()

    save(roots)
    print(f"[RS_{date}] FINAL {m/1e6:.2f}/{n/1e6:.2f} M, {len(subs)} subreddits ================")
    with open(f"{jsonl_dir}/readme_roots.txt", "a", encoding="utf-8") as f:
        f.write(f"[{date}] saved {m}/{n}\n")


def extract_txt(sub, year, tokenizer, overwrite=False, max_subword=3):
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

        ids = tokenizer.encode(txt)
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
                if m % 1e4 == 0:
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
                    with open(path_out, "a", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    lines = []
    if lines:
        with open(path_out, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    s = f"[{sub} {year}] txt kept {m}/{n}"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def extract_trees(sub, year):
    dir = f"{redditsub_dir}/{sub}"
    os.makedirs(dir, exist_ok=True)
    path_out = f"{dir}/{year}_trees.pkl"
    if os.path.exists(path_out):
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

    print(f"[{sub} {year}] {len(trees)} trees {n/len(trees):.1f} nodes/tree")
    os.makedirs(dir, exist_ok=True)
    pickle.dump(trees, open(path_out, "wb"))


def extract_time(sub, year, overwrite=False):
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

    s = f"[{sub} {year}] time kept {m}/{n}"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def calc_feedback(sub, year, overwrite=False):
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
        print("empty updown:")
        return

    with open(path_out, "w", encoding="utf-8") as f:
        f.write("\t".join(["#path", "vol", "width", "depth", "updown"]) + "\n")

    print(f"[{sub} {year}] calculating scores for {len(trees)} trees...")

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
        s = f"[{sub} {year}] {n_tree} tree {n_node} nodes"
    else:
        s = f"[{sub} {year}] trees are empty!"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def create_pairs(year, sub, feedback, overwrite=False):
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
    print(f"[{sub} {year}] creating pairs...")

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

    s = f"[{sub} {year} {feedback}] {n_line} pairs"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def add_seq(sub, year, feedback, overwrite=False):
    fname = f"{year}_{feedback}"
    dir = f"{redditsub_dir}/{sub}"
    turn_sep = " 50256 "
    path_out = f"{dir}/{fname}_ids.tsv"
    path_done = f"{path_out}.done"
    path = f"{dir}/{fname}.tsv"

    if os.path.exists(path_done) and not overwrite:
        return
    if not os.path.exists(path):
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
    print(f"[{sub} {year} {feedback}] adding seq")

    lines = []
    n = 0
    m = 0
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

    s = f"[{sub} {year} {feedback}] pair seq {m}/{n}"
    with open(path_done, "w") as f:
        f.write(s)
    print(s)


def combine_sub(year, feedback, overwrite=False, skip_same_pos=True):
    dir = f"{output_dir}/{feedback}"
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


def split_by_root(path, p_test=0.01):
    print("spliting by root " + path)
    lines = {
        "train": [],
        "vali": [],
    }
    prev = None
    n = 0

    for k in lines:
        if len(lines[k]) == 0:
            continue
        open(path + "." + k, "w", encoding="utf-8")

    for line in open(path, "r", encoding="utf-8"):
        line = line.strip("\n")
        if not line:
            continue
        cxt = line.split("\t")[3]
        root = cxt.strip().split()[0]
        if root != prev:
            if np.random.random() < p_test:
                k = "vali"
            else:
                k = "train"
        # pdb.set_trace()
        lines[k].append(line)
        prev = root
        n += 1
        if n % 1e6 == 0:
            print("read %i M" % (n / 1e6))
            for k in lines:
                if len(lines[k]) == 0:
                    continue
                with open(path + "." + k, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines[k]) + "\n")
                lines[k] = []

    for k in lines:
        if len(lines[k]) == 0:
            continue
        with open(path + "." + k, "a", encoding="utf-8") as f:
            f.write("\n".join(lines[k]))
        lines[k] = []


def shuffle(feedback, part, n_temp=10):
    dir = f"{output_dir}/{feedback}"
    path = f"{dir}/raw.tsv.{part}"
    path_out = f"{dir}/{part}.tsv"
    dir_temp = f"{output_dir}/temp/{feedback}"

    print("slicing " + path)
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
    print(dir_temp)
    for i in range(n_temp):
        print(f"reading temp{i}")
        lines = open(f"{dir_temp}/temp{i}", "r", encoding="utf-8").readlines()
        print("shuffling")
        jj = list(range(len(lines)))
        np.random.shuffle(jj)
        print("writing")
        with open(path_out, "a", encoding="utf-8") as f:
            f.write("\n".join([lines[j].strip("\n") for j in jj]) + "\n")


def get_subs():
    return ["4chan"]
    # print("collectiing subs...")
    # subs = sorted(os.listdir(redditsub_dir))
    # print("collected %i subs" % len(subs))
    # return subs


def build_json(year, is_extracted=False):
    for date in get_dates(year):
        extract_rc(date, is_extracted=False)
        extract_rs(date, is_extracted=False)


def build_basic(year):
    from transformers19 import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    subs = get_subs()
    for sub in subs:
        extract_time(sub, year)
        extract_txt(sub, year, tokenizer)
        extract_trees(sub, year)
        calc_feedback(sub, year, overwrite=False)


def build_pairs(year, feedback):
    subs = get_subs()
    for sub in subs:
        create_pairs(year, sub, feedback, overwrite=False)
        add_seq(sub, year, feedback, overwrite=False)
    path = combine_sub(year, feedback)
    split_by_root(path)
    for part in ["train", "vali"]:
        shuffle(feedback, part)


def main():
    year = 2011
    build_json(year, is_extracted=False)
    build_basic(year)

    tasks = ["updown", "depth", "width"]
    for t in tasks:
        build_pairs(year, t)


if __name__ == "__main__":
    main()
