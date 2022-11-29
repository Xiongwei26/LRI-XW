"""
作者：XWEI
日期：2022年10月02日
"""
from sklearn.decomposition import KernelPCA, NMF
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA


def KPCA(feature):
    clf = KernelPCA(n_components=400)
    new_feature = clf.fit_transform(feature)
    return new_feature


def LLE(feature):
    clf = LocallyLinearEmbedding(n_components=400)
    new_feature = clf.fit_transform(feature)
    return new_feature


def dim_NMF(feature):
    clf = NMF(n_components=400)
    new_feature = clf.fit_transform(feature)
    return new_feature
