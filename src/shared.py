# author: Xiang Gao at Microsoft Research AI NLP Group
import shlex
import os, subprocess

_cat_ = ' <-COL-> '
#EOS_token = '_EOS_'   # old version, before Nov 8 2020
EOS_token = '<|endoftext|>'


def download_model(path, overwrite=False):
    if path is None:
        return
    links = dict()
    for k in ['updown', 'depth', 'width', 'human_vs_rand', 'human_vs_machine']:
        if os.path.exists(f"{path}/{k}.pth"):
            if overwrite == False:
                print("WARNING: Path to the model already exists. Skipping.")
                continue
            else:
                print("WARNING: Path to the model already exists. Overwriting.")
        links['%s.pth'%k] = 'https://xiagnlp2.blob.core.windows.net/dialogrpt/%s.pth'%k
    links['medium_ft.pkl'] = 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl'
    print(links)
    cmds = [f"wget -q -P '{path}' {links[model_name]}" for model_name in links ]
    print("Running these commands:")
    print('\n'.join(cmds))
    processes = [subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE) for cmd in cmds]
    [process.communicate() for process in processes]
