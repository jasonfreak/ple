#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from feature import Feature

def doWithOneHotEncoder(model, featureList):
    assert(isinstance(model, OneHotEncoder))
    assert(hasattr(model, 'feature_indices_'))

    leaves = np.array([])

    n_features = len(featureList)
    
    if model.categorical_features == 'all':
        mask_features = np.ones(n_features)
    else:
        mask_features = np.zeros(n_features)
        mask_features[self.categorical_features] = 1

    n_qualitativeFeatures = len(model.feature_indices_) - 1
    if model.n_values == 'auto':
        n_activeFeatures = len(model.active_features_)
    j = k = 0
    for i in range(n_features):
        feature = featureList[i]
        if mask_features[i]:
            if model.n_values == 'auto':
                while k < n_activeFeatures and model.active_features_[k] < model.feature_indices_[j+1]:
                    newFeature = Feature(feature.name)
                    feature.transform('%s[%d]' % (model.__class__.__name__, model.active_features_[k] - model.feature_indices_[j]), newFeature)
                    leaves = np.append(leaves, newFeature)
                    k += 1
            else:
                for k in range(model.feature_indices_[j]+1, model.feature_indices_[j+1]):
                    newFeature = Feature(feature.name)
                    feature.transform('%s[%d]' % (model.__class__.__name__, k - model.feature_indices_[j]), newFeature)
                    leaves = np.append(leaves, newFeature)
            j += 1
        else:
            newFeature = Feature(feature.name)
            feature.transform('%s[r]' % model.__class__.__name__, newFeature)
            leaves = append(leaves, newFeatures)

    return leaves

def main():
    X = [[1, 2], [2, 3]]
    
    root = Feature('root')
    featureList = np.array([])
    for i in range(len(X[0])):
        feature = Feature('feature_%d' % i)
        root.transform('init', feature)
        featureList = np.append(featureList, feature)

    model = OneHotEncoder(n_values=[5,8], sparse=True)
    model.fit(X)
    doWithOneHotEncoder(model, featureList)
    root.printTree()

if __name__ == '__main__':
    main()
