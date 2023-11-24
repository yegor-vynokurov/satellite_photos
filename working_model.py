import numpy as np 
import pandas as pd
import geopandas as gpd

import os, fnmatch

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize

from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union, unary_union

import matplotlib.pyplot as plt

# If we are on colab: this clones the repo and installs the dependencies

from pathlib import Path

if Path.cwd().name != "LightGlue":
    !git clone --quiet https://github.com/cvg/LightGlue/
    %cd LightGlue
    !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device) # load the matcher

path1 = input('enter a path to a first image') # input a path to image for matching
path2 = input('enter a path to a second image') # input a path to image for matching

        
image0 = load_image(path1) # read first image for keypoints matching
image1 = load_image(path2) # read second image for keypoints matching

feats0 = extractor.extract(image0.to(device)) # extract keypoints from a first image
feats1 = extractor.extract(image1.to(device)) # extract keypoints from a second image
matches01 = matcher({"image0": feats0, "image1": feats1}) # matching of images
feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]] # select matched keypoints

axes = viz2d.plot_images([image0, image1]) # visualize images
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2) # visualize matches
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1]) # visualize images
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10) # visualize matched and no-matched keypoints
plt.show()