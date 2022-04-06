import os
from glob import glob
from time import strftime
from google.colab import files


def copy_weights_files(folder, dest):
    fnames = glob(f"{folder}darknet/backup/*.weights")
    max_epoch = 0
    for i, fname in enumerate(fnames):
        if "best" in fname or "last" in fname:
            os.system(f"cp {fname} {dest}")
        else:
            try:
                epoch = int(fname.split("/")[-1].split("_")[-1].replace(".weights", ""))
                if epoch > max_epoch:
                    max_epoch = epoch
                    fname_max_epoch = fnames[i]
            except ValueError:
                continue
    os.system(f"cp {fname_max_epoch} {dest}")


def download_artefacts(folder):
    dest = f"{folder}model_files/"
    copy_weights_files(folder, dest)
    os.system(f"cp {f'{folder}darknet/chart.png'} {dest}")
    out_fname = f"{folder}model_files_{strftime('%Y%m%d-%H%M')}.zip"
    os.system(f"zip -r {out_fname} {dest}")
    files.download(out_fname)
