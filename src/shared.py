# author: Xiang Gao at Microsoft Research AI NLP Group
import shlex


_cat_ = ' <-COL-> '
#EOS_token = '_EOS_'   # old version, before Nov 8 2020
EOS_token = '<|endoftext|>'


def download_model(path):
    if path is None:
        return
    import os, subprocess
    if os.path.exists(path):
        print("WARNING: Path to the model already exists. Overwriting.")
    links = dict()
    for k in ['updown', 'depth', 'width', 'human_vs_rand', 'human_vs_machine']:
        links['%s.pth'%k] = 'https://xiagnlp2.blob.core.windows.net/dialogrpt/%s.pth'%k
    links['medium_ft.pkl'] = 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl'
    print(links)
    cmds = [f"wget -P '{path}' {links[model_name]}" for model_name in links ]
    print("Running these commands:")
    print('\n'.join(cmds))
    processes = [subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE) for cmd in cmds]
    [process.communicate() for process in processes]
