import time
import os
import sys

import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_num_threads(1)
torch.set_printoptions(
    precision=10,
    threshold=sys.maxsize,
    linewidth=2**14,
    sci_mode=False
)

def heatmap(data, x_labels, y_labels, fname, shift=False):
    if shift:
        shifted = torch.empty_like(data)
        shifted[0, :] = data[0, :]
        shifted[:, 0] = data[:, 0]
        shifted[1:, 1:] = data[:-1, :-1]
        data = shifted
 
    plt.clf()

    # Modify to look better for different sequnce lengths
    cell_dim = 0.4
    fontsize = 16

    fontname = "Nimbus Roman"
    cmap = "inferno"

    nrows, ncols = data.shape
    fig_width = ncols * cell_dim
    fig_height = nrows * cell_dim

    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(data.cpu(), cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=plt.gca(), fraction=0.04, pad=0.04)

    plt.xlabel("Teacher tokens", fontname=fontname, fontsize=(fontsize + 2), fontweight="bold")
    plt.ylabel("Student tokens", fontname=fontname, fontsize=(fontsize + 2), fontweight="bold")
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, fontname=fontname, fontsize=fontsize, rotation="vertical")
    plt.yticks(ticks=range(len(y_labels)), labels=y_labels, fontname=fontname, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(fname=fname)


def analyse_main(save_dir, num_heatmaps=None):
    jsonl_fname = os.path.join(save_dir, "align.jsonl")
    heatmap_dir = os.path.join(save_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    with open(jsonl_fname, "r") as f:
        for i, line in tqdm(enumerate(f), desc=f"Saving "):
            if num_heatmaps is not None and i >= num_heatmaps:
                break

            obj = json.loads(line)
            s_masked = obj["student"]
            t_masked = obj["teacher"]
            align_masked = torch.tensor(obj["alignment"])

            heatmap(data=align_masked, x_labels=t_masked, y_labels=s_masked, fname=os.path.join(heatmap_dir, f"align_{i}.jpg"), shift=(not "chunk" in save_dir))

def main():
    # Command-line arguments: save directory; number of heatmaps to produce (optional)
    save_dir = sys.argv[1]
    num_heatmaps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    analyse_main(save_dir, num_heatmaps)

if __name__ == "__main__":
    main()