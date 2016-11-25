import csv
import numpy as np
import pylab
from skimage import transform, filter, color
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer


def loadTags(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)

    tagName = [r[0] for r in data]
    return tagName, dict(zip(tagName, range(len(tagName))))

def getTagScore(scores, tags, tag2IDs):
    scores = np.exp(scores)
    scores /= scores.sum()
    tagScore = []
    for r in tags:
        tagScore.append((r, scores[tag2IDs[r]]))

    return tagScore

def showAttMap(img, attMaps, tagName, overlap = True, blur = False, save = False):
    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    f, ax = plt.subplots(int(len(tagName)/2+1), 2)
    ax[0,0].imshow(img)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = color.gray2rgb(img)

    for i in range(len(tagName)):
        attMap = attMaps[i].copy()
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
        if blur:
            attMap = filter.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()

        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
#        Tracer()()
        if overlap:
            attMap = 1*(1-attMap**0.8).reshape(attMap.shape + (1,))*img + (attMap**0.8).reshape(attMap.shape+(1,)) * attMapV;


        ax[(i+1)/2, (i+1)%2].imshow(attMap, interpolation = 'bicubic')
        ax[(i+1)/2, (i+1)%2].set_title(tagName[i])

    if save:
        f.savefig("attentionmaps.jpg")

