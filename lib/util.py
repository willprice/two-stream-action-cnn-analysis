import csv
import numpy as np
import pylab
from skimage import transform, filter, color
import matplotlib.pyplot as plt


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

def showAttMap(img, attMaps, tagName, overlap = True, cmap='autumn', blur = False, save = False, ):
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

        cmap = plt.get_cmap(cmap)
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1*(1-attMap**0.8).reshape(attMap.shape + (1,))*img + (attMap**0.8).reshape(attMap.shape+(1,)) * attMapV;


        current_subplot_axes = ax[int((i+1)/2), int((i+1)%2)]
        attention_map_subplot = current_subplot_axes.imshow(attMap, interpolation = 'bicubic', cmap=cmap)
        plt.colorbar(attention_map_subplot, ax=current_subplot_axes)
        current_subplot_axes.set_title(tagName[i])

    if save:
        f.savefig("attentionmaps.jpg")
