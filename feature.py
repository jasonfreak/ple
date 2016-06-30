import numpy as np

class Transform(object):
    def __init__(self, label, feature):
        super(Transform, self).__init__()
        self.label = label
        self.feature = feature

class Feature(object):
    def __init__(self, name):
        super(Feature, self).__init__()
        self.name = name
        self.label = '%s[%d]' % (self.name, id(self))
        self.transformList = np.array([])

    def transform(self, label, feature):
        self.transformList = np.append(self.transformList, Transform(label, feature))

    def printTree(self):
        print self.label
        for transform in self.transformList:
            feature = transform.feature
            print '--%s-->' % transform.label,
            feature.printTree()

    def __str__(self):
        return self.label
