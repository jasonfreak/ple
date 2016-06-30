#!/usr/bin/env python
import numpy as np
from sklearn.feature_selection.base import SelectorMixin
from feature import Feature

def doWithSelector(model, featureList):
    assert(isinstance(model, SelectorMixin))

    leaves = np.array([])

    n_features = len(featureList)
    
    mask_features = model.get_support()

    for i in range(n_features):
        feature = featureList[i]
        if mask_features[i]:
            newFeature = Feature(feature.name)
            feature.transform(model.__class__.__name__, newFeature)
            leaves = np.append(leaves, newFeature)
        else:
            newFeature = Feature('Abandomed')
            feature.transform(model.__class__.__name__, newFeature)

    return leaves

def main():
    from sklearn.feature_selection import VarianceThreshold
    X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
    
    root = Feature('root')
    featureList = np.array([])
    for i in range(len(X[0])):
        feature = Feature('feature_%d' % i)
        root.transform('init', feature)
        featureList = np.append(featureList, feature)

    model = VarianceThreshold()
    model.fit(X)
    doWithSelector(model, featureList)
    root.printTree()

if __name__ == '__main__':
    main()
