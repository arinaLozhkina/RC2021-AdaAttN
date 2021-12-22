from PIL import Image
import numpy as np
from os import listdir
from sys import argv
import seaborn as sns
import matplotlib.pyplot as plt

dirname = "AddaAttN_out"
out_ext = "png"
chi2_distance = lambda a, b: 0.5 * np.sum([((a - b) ** 2) / ((a + b) + 1e-20)])
fig, ax = plt.subplots(nrows=3, figsize=(12, 7), sharex=True)

s = listdir("style")[0]
c = listdir("content")[0]
cs_name = f"{s[:-4]}_{c[:-4]}.{out_ext}"
print(cs_name)
cs_image = np.array(Image.open(f"{dirname}/{cs_name}").convert('RGB'))
style_image = np.array(Image.open(f"style/{s}").convert('RGB'))
chi2_distance = lambda a, b: 0.5 * np.sum([((a - b) ** 2) / ((a + b) + 1e-20)])
dist = []
sp = 1
rgb = ["red", "green", "blue"]
for i in range(3):

    h, bin_edges = np.histogram(cs_image[:, :, i], bins=np.arange(0,256), density=True)
    ax[i].bar(bin_edges[:-1],h, color="#F8B195")
    h2, bin_edges2 = np.histogram(style_image[:, :, i], bins=np.arange(0, 256), density=True)
    ax[i].bar(bin_edges2[:-1],h2, alpha=0.3, color=rgb[i])
    ax[i].set_ylabel(rgb[i])
    dist.append(chi2_distance(h, h2))
    plt.tight_layout()
    if i == 2:
        ax[i].set_xlabel("pixel value")
print(np.mean(dist))

