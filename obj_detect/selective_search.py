import torch
from torch_snippets import *
import selectivesearch
from skimage.segmentation import felzenszwalb
import cv2

# img = read('image.jpg', 1)
# segments = felzenszwalb(img, scale=200)
# subplots([img, segments], titles=['Original image', 'after segmentation'], sz=10, nc=2)

def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=1000)
    print(img_lbl.shape)
    img_area = np.prod(img.shape[:2])
    candidates = []

    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] in candidates: continue
        if r['size'] > (1 * img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates

img = read('image.jpg', 1)
candidates = extract_candidates(img)
show(img, bbs=candidates)