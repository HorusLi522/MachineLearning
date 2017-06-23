from sklearn.linear_model.bayes import ARDRegression
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.cluster.affinity_propagation_ import AffinityPropagation
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.mixture.bayesian_mixture import BayesianGaussianMixture
from sklearn.linear_model.bayes import BayesianRidge
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network.rbm import BernoulliRBM
from sklearn.preprocessing.data import Binarizer
from sklearn.cluster.birch import Birch
from sklearn.cross_decomposition.cca_ import CCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.mixture.dpgmm import DPGMM
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.decomposition.dict_learning import DictionaryLearning
from sklearn.linear_model.coordinate_descent import ElasticNet
from sklearn.linear_model.coordinate_descent import ElasticNetCV
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance
from sklearn.tree.tree import ExtraTreeClassifier
from sklearn.tree.tree import ExtraTreeRegressor
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.decomposition.factor_analysis import FactorAnalysis
from sklearn.decomposition.fastica_ import FastICA
from sklearn.cluster.hierarchical import FeatureAgglomeration
from sklearn.preprocessing._function_transformer import FunctionTransformer
from sklearn.mixture.gmm import GMM
from sklearn.mixture.gaussian_mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.gaussian_process import GaussianProcess
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection.univariate_selection import GenericUnivariateSelect
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.covariance.graph_lasso_ import GraphLasso
from sklearn.covariance.graph_lasso_ import GraphLassoCV
from sklearn.linear_model.huber import HuberRegressor
from sklearn.preprocessing.imputation import Imputer
from sklearn.decomposition.incremental_pca import IncrementalPCA
from sklearn.ensemble.iforest import IsolationForest
from sklearn.manifold.isomap import Isomap
from sklearn.cluster.k_means_ import KMeans
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.preprocessing.data import KernelCenterer
from sklearn.neighbors.kde import KernelDensity
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors.approximate import LSHForest
from sklearn.semi_supervised.label_propagation import LabelPropagation
from sklearn.semi_supervised.label_propagation import LabelSpreading
from sklearn.linear_model.least_angle import Lars
from sklearn.linear_model.least_angle import LarsCV
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.linear_model.coordinate_descent import LassoCV
from sklearn.linear_model.least_angle import LassoLars
from sklearn.linear_model.least_angle import LassoLarsCV
from sklearn.linear_model.least_angle import LassoLarsIC
from sklearn.decomposition.online_lda import LatentDirichletAllocation
from sklearn.covariance.shrunk_covariance_ import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model.base import LinearRegression
from sklearn.svm.classes import LinearSVC
from sklearn.svm.classes import LinearSVR
from sklearn.manifold.locally_linear import LocallyLinearEmbedding
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.logistic import LogisticRegressionCV
from sklearn.manifold.mds import MDS
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.preprocessing.data import MaxAbsScaler
from sklearn.cluster.mean_shift_ import MeanShift
from sklearn.covariance.robust_covariance import MinCovDet
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.decomposition.dict_learning import MiniBatchDictionaryLearning
from sklearn.cluster.k_means_ import MiniBatchKMeans
from sklearn.decomposition.sparse_pca import MiniBatchSparsePCA
from sklearn.linear_model.coordinate_descent import MultiTaskElasticNet
from sklearn.linear_model.coordinate_descent import MultiTaskElasticNetCV
from sklearn.linear_model.coordinate_descent import MultiTaskLasso
from sklearn.linear_model.coordinate_descent import MultiTaskLassoCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition.nmf import NMF
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.preprocessing.data import Normalizer
from sklearn.svm.classes import NuSVC
from sklearn.svm.classes import NuSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.covariance.shrunk_covariance_ import OAS
from sklearn.svm.classes import OneClassSVM
from sklearn.linear_model.omp import OrthogonalMatchingPursuit
from sklearn.linear_model.omp import OrthogonalMatchingPursuitCV
from sklearn.decomposition.pca import PCA
from sklearn.cross_decomposition.pls_ import PLSCanonical
from sklearn.cross_decomposition.pls_ import PLSRegression
from sklearn.cross_decomposition.pls_ import PLSSVD
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor
from sklearn.linear_model.perceptron import Perceptron
from sklearn.decomposition.nmf import ProjectedGradientNMF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.regression import RadiusNeighborsRegressor
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model.randomized_l1 import RandomizedLasso
from sklearn.linear_model.randomized_l1 import RandomizedLogisticRegression
from sklearn.decomposition.pca import RandomizedPCA
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.preprocessing.data import RobustScaler
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.svm.classes import SVC
from sklearn.svm.classes import SVR
from sklearn.feature_selection.univariate_selection import SelectFdr
from sklearn.feature_selection.univariate_selection import SelectFpr
from sklearn.feature_selection.univariate_selection import SelectFwe
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.covariance.shrunk_covariance_ import ShrunkCovariance
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.decomposition.sparse_pca import SparsePCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.spectral import SpectralClustering
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.manifold.spectral_embedding_ import SpectralEmbedding
from sklearn.preprocessing.data import StandardScaler
from sklearn.manifold.t_sne import TSNE
from sklearn.linear_model.theil_sen import TheilSenRegressor
from sklearn.mixture.dpgmm import VBGMM
from sklearn.feature_selection.variance_threshold import VarianceThreshold

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


clf_dict = {'ARDRegression':ARDRegression(),
			'AdaBoostClassifier':AdaBoostClassifier(),
			'AdaBoostRegressor':AdaBoostRegressor(),
			'AdditiveChi2Sampler':AdditiveChi2Sampler(),
			'AffinityPropagation':AffinityPropagation(),
			'AgglomerativeClustering':AgglomerativeClustering(),
			'BaggingClassifier':BaggingClassifier(),
			'BaggingRegressor':BaggingRegressor(),
			'BayesianGaussianMixture':BayesianGaussianMixture(),
			'BayesianRidge':BayesianRidge(),
			'BernoulliNB':BernoulliNB(),
			'BernoulliRBM':BernoulliRBM(),
			'Binarizer':Binarizer(),
			'Birch':Birch(),
			'CCA':CCA(),
			'CalibratedClassifierCV':CalibratedClassifierCV(),
			'DBSCAN':DBSCAN(),
			'DPGMM':DPGMM(),
			'DecisionTreeClassifier':DecisionTreeClassifier(),
			'DecisionTreeRegressor':DecisionTreeRegressor(),
			'DictionaryLearning':DictionaryLearning(),
			'ElasticNet':ElasticNet(),
			'ElasticNetCV':ElasticNetCV(),
			'EmpiricalCovariance':EmpiricalCovariance(),
			'ExtraTreeClassifier':ExtraTreeClassifier(),
			'ExtraTreeRegressor':ExtraTreeRegressor(),
			'ExtraTreesClassifier':ExtraTreesClassifier(),
			'ExtraTreesRegressor':ExtraTreesRegressor(),
			'FactorAnalysis':FactorAnalysis(),
			'FastICA':FastICA(),
			'FeatureAgglomeration':FeatureAgglomeration(),
			'FunctionTransformer':FunctionTransformer(),
			'GMM':GMM(),
			'GaussianMixture':GaussianMixture(),
			'GaussianNB':GaussianNB(),
			'GaussianProcess':GaussianProcess(),
			'GaussianProcessClassifier':GaussianProcessClassifier(),
			'GaussianProcessRegressor':GaussianProcessRegressor(),
			'GaussianRandomProjection':GaussianRandomProjection(),
			'GenericUnivariateSelect':GenericUnivariateSelect(),
			'GradientBoostingClassifier':GradientBoostingClassifier(),
			'GradientBoostingRegressor':GradientBoostingRegressor(),
			'GraphLasso':GraphLasso(),
			'GraphLassoCV':GraphLassoCV(),
			'HuberRegressor':HuberRegressor(),
			'Imputer':Imputer(),
			'IncrementalPCA':IncrementalPCA(),
			'IsolationForest':IsolationForest(),
			'Isomap':Isomap(),
			'KMeans':KMeans(),
			'KNeighborsClassifier':KNeighborsClassifier(),
			'KNeighborsRegressor':KNeighborsRegressor(),
			'KernelCenterer':KernelCenterer(),
			'KernelDensity':KernelDensity(),
			'KernelPCA':KernelPCA(),
			'KernelRidge':KernelRidge(),
			'LSHForest':LSHForest(),
			'LabelPropagation':LabelPropagation(),
			'LabelSpreading':LabelSpreading(),
			'Lars':Lars(),
			'LarsCV':LarsCV(),
			'Lasso':Lasso(),
			'LassoCV':LassoCV(),
			'LassoLars':LassoLars(),
			'LassoLarsCV':LassoLarsCV(),
			'LassoLarsIC':LassoLarsIC(),
			'LatentDirichletAllocation':LatentDirichletAllocation(),
			'LedoitWolf':LedoitWolf(),
			'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis(),
			'LinearRegression':LinearRegression(),
			'LinearSVC':LinearSVC(),
			'LinearSVR':LinearSVR(),
			'LocallyLinearEmbedding':LocallyLinearEmbedding(),
			'LogisticRegression':LogisticRegression(),
			'LogisticRegressionCV':LogisticRegressionCV(),
			'MDS':MDS(),
			'MLPClassifier':MLPClassifier(),
			'MLPRegressor':MLPRegressor(),
			'MaxAbsScaler':MaxAbsScaler(),
			'MeanShift':MeanShift(),
			'MinCovDet':MinCovDet(),
			'MinMaxScaler':MinMaxScaler(),
			'MiniBatchDictionaryLearning':MiniBatchDictionaryLearning(),
			'MiniBatchKMeans':MiniBatchKMeans(),
			'MiniBatchSparsePCA':MiniBatchSparsePCA(),
			'MultiTaskElasticNet':MultiTaskElasticNet(),
			'MultiTaskElasticNetCV':MultiTaskElasticNetCV(),
			'MultiTaskLasso':MultiTaskLasso(),
			'MultiTaskLassoCV':MultiTaskLassoCV(),
			'MultinomialNB':MultinomialNB(),
			'NMF':NMF(),
			'NearestCentroid':NearestCentroid(),
			'NearestNeighbors':NearestNeighbors(),
			'Normalizer':Normalizer(),
			'NuSVC':NuSVC(),
			'NuSVR':NuSVR(),
			'Nystroem':Nystroem(),
			'OAS':OAS(),
			'OneClassSVM':OneClassSVM(),
			'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit(),
			'OrthogonalMatchingPursuitCV':OrthogonalMatchingPursuitCV(),
			'PCA':PCA(),
			'PLSCanonical':PLSCanonical(),
			'PLSRegression':PLSRegression(),
			'PLSSVD':PLSSVD(),
			'PassiveAggressiveClassifier':PassiveAggressiveClassifier(),
			'PassiveAggressiveRegressor':PassiveAggressiveRegressor(),
			'Perceptron':Perceptron(),
			'ProjectedGradientNMF':ProjectedGradientNMF(),
			'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis(),
			'RANSACRegressor':RANSACRegressor(),
			'RBFSampler':RBFSampler(),
			'RadiusNeighborsClassifier':RadiusNeighborsClassifier(),
			'RadiusNeighborsRegressor':RadiusNeighborsRegressor(),
			'RandomForestClassifier':RandomForestClassifier(),
			'RandomForestRegressor':RandomForestRegressor(),
			'RandomizedLasso':RandomizedLasso(),
			'RandomizedLogisticRegression':RandomizedLogisticRegression(),
			'RandomizedPCA':RandomizedPCA(),
			'Ridge':Ridge(),
			'RidgeCV':RidgeCV(),
			'RidgeClassifier':RidgeClassifier(),
			'RidgeClassifierCV':RidgeClassifierCV(),
			'RobustScaler':RobustScaler(),
			'SGDClassifier':SGDClassifier(),
			'SGDRegressor':SGDRegressor(),
			'SVC':SVC(),
			'SVR':SVR(),
			'SelectFdr':SelectFdr(),
			'SelectFpr':SelectFpr(),
			'SelectFwe':SelectFwe(),
			'SelectKBest':SelectKBest(),
			'SelectPercentile':SelectPercentile(),
			'ShrunkCovariance':ShrunkCovariance(),
			'SkewedChi2Sampler':SkewedChi2Sampler(),
			'SparsePCA':SparsePCA(),
			'SparseRandomProjection':SparseRandomProjection(),
			'SpectralBiclustering':SpectralBiclustering(),
			'SpectralClustering':SpectralClustering(),
			'SpectralCoclustering':SpectralCoclustering(),
			'SpectralEmbedding':SpectralEmbedding(),
			'StandardScaler':StandardScaler(),
			'TSNE':TSNE(),
			'TheilSenRegressor':TheilSenRegressor(),
			'VBGMM':VBGMM(),
			'VarianceThreshold':VarianceThreshold(),}

    
