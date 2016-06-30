from sklearn.feature_selection.base import SelectorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one 
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from default import doWithDefault
from one2one import doWithSelector
from one2many import doWithOneHotEncoder
from many2many import doWithPCA
from feature import Feature

class PipelineExt(Pipeline):
    def _pre_get_featues(self, featureList):
        leaves = featureList
        for name, transform in self.steps[:-1]:
            leaves = _doWithModel(transform, leaves)
        return leaves

    def getFeatureList(self, featureList):
        leaves = self._pre_get_featues(featureList)
        model = self.steps[-1][-1]
        if hasattr(model, 'fit_transform') or hasattr(model, 'transform'):
            leaves = _doWithModel(model, leaves)
        return leaves

class FeatureUnionExt(FeatureUnion):
    def __init__(self, transformer_list, idx_list, n_jobs=1, transformer_weights=None):
        self.idx_list = idx_list
        FeatureUnion.__init__(self, transformer_list=map(lambda trans:(trans[0], trans[1]), transformer_list), n_jobs=n_jobs, transformer_weights=transformer_weights)

    def fit(self, X, y=None):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X[:,idx], y)
            for name, trans, idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, X[:,idx], y,
                                        self.transformer_weights, **fit_params)
            for name, trans, idx in transformer_idx_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def transform(self, X):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, X[:,idx], self.transformer_weights)
            for name, trans, idx in transformer_idx_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def getFeatureList(self, featureList):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        leaves = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(_doWithModel)(trans, featureList[idx])
            for name, trans, idx in transformer_idx_list))
        leaves = np.hstack(leaves)
        return leaves

def _doWithModel(model, featureList):
    if isinstance(model, SelectorMixin):
        return doWithSelector(model, featureList)
    elif isinstance(model, OneHotEncoder):
        return doWithOneHotEncoder(model, featureList)
    elif isinstance(model, PCA):
        return doWithPCA(model, featureList)
    elif isinstance(model, FeatureUnionExt) or isinstance(model, PipelineExt):
        return model.getFeatureList(featureList)
    else:
        return doWithDefault(model, featureList)

def initRoot(featureNameList):
    root = Feature('root')
    for featureName in featureNameList:
        newFeature = Feature(featureName)
        root.transform('init', newFeature)
    return root

def _draw(G, root, edgeLabelDict):
    for transform in root.transformList:
        G.add_edge(root.label, transform.feature.label)
        edgeLabelDict[(root.label, transform.feature.label)] = transform.label
        _draw(G, transform.feature, edgeLabelDict)

def draw(root):
    G = nx.DiGraph()
    edgeLabelDict = {}

    _draw(G, root, edgeLabelDict)
#    pos=nx.circular_layout(G)
#    pos=nx.spectral_layout(G)
#    pos=nx.spring_layout(G)
#    pos=nx.spring_layout(G, pos=pos)
    pos=nx.spring_layout(G, iterations=150)
#    pos=nx.circular_layout(G)
#    pos=nx.fruchterman_reingold_layout(G)
#    pos=nx.random_layout(G)
#    pos=nx.shell_layout(G)

    nx.draw_networkx_nodes(G,pos,node_size=50, node_color="white")
    nx.draw_networkx_edges(G,pos, width=1,alpha=0.5,edge_color='black')
    nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, edgeLabelDict)

    plt.show()
