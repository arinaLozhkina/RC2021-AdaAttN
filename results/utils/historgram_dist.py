from PIL import Image
import numpy as np
from os import listdir
from sys import argv

assert len(argv) == 4

dirname = argv[1]
output = argv[2]
out_ext = argv[3]
chi2_distance = lambda a, b: 0.5 * np.sum([((a - b) ** 2) / ((a + b) + 1e-20)])

with open(output, "w") as f:
    for s in listdir("style"):
        for c in listdir("content"):
            cs_name = f"{s[:-4]}_{c[:-4]}.{out_ext}"
            
            cs_image = np.array(Image.open(f"{dirname}/{cs_name}").convert('RGB'))
            style_image = np.array(Image.open(f"style/{s}").convert('RGB'))
            chi2_distance = lambda a, b: 0.5 * np.sum([((a - b) ** 2) / ((a + b) + 1e-20)])
            dist = []
            for i in range(3):
                hist, bin_edges = np.histogram(cs_image[:, :, i], bins=np.arange(0,256), density=True)
                hist2, bin_edges2 = np.histogram(style_image[:, :, i], bins=np.arange(0,256), density=True)
                dist.append(chi2_distance(hist, hist2))

            f.write(f'"{cs_name}": {np.mean(dist)}\n')
            
