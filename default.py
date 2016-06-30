#!/usr/bin/env python
import numpy as np
from feature import Feature

def doWithDefault(model, featureList):
    leaves = np.array([])

    n_features = len(featureList)
    
    for i in range(n_features):
        feature = featureList[i]
        newFeature = Feature(feature.name)
        feature.transform(model.__class__.__name__, newFeature)
        leaves = np.append(leaves, newFeature)

    return leaves


def main():
    pass

if __name__ == '__main__':
    main()
