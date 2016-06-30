#!/usr/bin/env python
import numpy as np
from sklearn.decomposition import PCA
from feature import Feature

def doWithPCA(model, featureList):
    leaves = np.array([])

    n_features = len(featureList)
    
    for i in range(model.n_components_):
        newFeature = Feature(model.__class__.__name__)
        leaves = np.append(leaves, newFeature)

    for i in range(n_features):
        feature = featureList[i]
        for j in range(model.n_components_):
            newFeature = leaves[j]
            feature.transform(model.__class__.__name__, newFeature)

    return leaves

def main():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    
    root = Feature('root')
    featureList = np.array([])
    for i in range(len(X[0])):
        feature = Feature('feature_%d' % i)
        root.transform('init', feature)
        featureList = np.append(featureList, feature)

    model = PCA(n_components=1)
    model.fit(X)
    doWithPCA(model, featureList)
    root.printTree()

if __name__ == '__main__':
    main()
