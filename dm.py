import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from ple import PipelineExt, FeatureUnionExt, initRoot, draw

def datamining(iris, featureList):
    step1 = ('Imputer', Imputer())
    step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
    step2_2 = ('ToLog', FunctionTransformer(np.log1p))
    step2_3 = ('ToBinary', Binarizer())
    step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))
    step3 = ('MinMaxScaler', MinMaxScaler())
    step4 = ('SelectKBest', SelectKBest(chi2, k=3))
    step5 = ('PCA', PCA(n_components=2))
    step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
    pipeline = PipelineExt(steps=[step1, step2, step3, step4, step5, step6])
    pipeline.fit(iris.data, iris.target)
    leaves = pipeline.getFeatureList(featureList)
    for i in range(len(leaves)):
        print leaves[i], pipeline.steps[-1][-1].coef_[i]

def main():
    iris = load_iris()
    iris.data = np.hstack((np.random.choice([0, 1, 2], size=iris.data.shape[0]+1).reshape(-1,1), np.vstack((iris.data, np.full(4, np.nan).reshape(1,-1)))))
    iris.target = np.hstack((iris.target, np.array([np.median(iris.target)])))
    root = initRoot(['color', 'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'])
    featureList = np.array([transform.feature for transform in root.transformList])

    datamining(iris, featureList)

    root.printTree()
    draw(root)

if __name__ == '__main__':
    main()

