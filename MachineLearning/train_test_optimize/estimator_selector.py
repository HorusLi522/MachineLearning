from classifier_dictionary import clf_dict

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.model_selection import GridSearchCV

from time import time

from pprint import pprint as p

class FilterClassifiers:
	'''
	Filters all sklearn classifiers by name 
	and attribute and prints their import statement
	and input for EstimatorSelectionHelper

	Params (from EstimatorSelectionHelper): 
		clf_type (str): str in classifier name
		feature	(str): str matching classifier attribute

	Example:
		clf_type = 'classifier'
		feature = 'feature_importances_'

		fc = FilterClassifiers(clf_type = clf_type, 
							   feature = feature)
	'''	
	def _filter(self):
		'''
		Adds to models list. Instatiated when class instantiation
		'''
		print("Filtered by {} and {}".format(self.clf_type,self.feature))

		for name, class_ in clf_dict.items():			
			if hasattr(class_, self.feature) and self.clf_type in name.lower():							
				if self.kwargs.get('clf_exclude'):
					if name in self.kwargs.get('clf_exclude')[0]:	
						print("Skipped {}".format(name))
						pass
					else:										
						print("Adding {} to clf queue".format(name))
						self._models.setdefault(class_,{}).update({})
				else:
					print("Adding {} to clf queue".format(name))
					self._models.setdefault(class_,{}).update({})
					
		if self.kwargs.get('clf_params'):
			for k,v in self.kwargs.get('clf_params').items():
				self._models.update({clf_dict[k]:v})      


class EstimatorSelectionHelper(FilterClassifiers):
	'''
	Uses clf and feature filters to find the best estimator

	Params (from EstimatorSelectionHelper): 
		clf_type (str): str in classifier name
		feature	(str): str matching classifier attribute
		kwargs (dict):
			CLF_EXCLUDE: parameter names to exclude 
			CLF_PARAMS: any additional clf params to be tested

	Example:

		from sklearn.datasets import load_iris
		from sklearn.model_selection import train_test_split

		data = load_iris()

		X = data.data
		y = data.target

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)		


		CLF_TYPE = 'class'
		FEATURE_SELECTION = 'predict_proba'
		VERBOSE = False
		CLF_EXCLUDE = ['ExtraTreeClassifier',
					   'AdaBoostClassifier',
					   'RandomForestClassifier',
					   # 'ExtraTreesClassifier',
					   # 'DecisionTreeClassifier',
					   # 'MLPClassifier',							  											  				
					   # 'KNeighborsClassifier',
					   # 'BaggingClassifier',
					   # 'GaussianProcessClassifier',
					   # 'GradientBoostingClassifier',
					   # 'CalibratedClassifierCV',
					   # 'PassiveAggressiveClassifier'
					   ],
		CLF_PARAMS = {
		  			  'PassiveAggressiveClassifier':{
													 'random_state':[0,1],
													 'n_iter':[2,12,2]},
					   'CalibratedClassifierCV':{'method':['sigmoid',
					   									   'isotonic']}											
					  }

		fc = EstimatorSelectionHelper(clf_type=CLF_TYPE, 
									  feature= FEATURE_SELECTION,	
									  verbose=VERBOSE,						  
									  clf_exclude=CLF_EXCLUDE,
									  clf_params=CLF_PARAMS)

		fc.fit(X=X, y=y)
		fc.estimator_selector()
		summary = fc.score_summary()

		>>>summary
			Filtered by class and predict_proba
			Adding KNeighborsClassifier to clf queue
			...
			Running GridSearchCV for MLPClassifier.
			                          name     score                         params exec_time
			8                MLPClassifier  0.986667                           None    0.905s
			...
			2  PassiveAggressiveClassifier  0.780000  [random_state: 0, n_iter: 12]    0.069s

		>>>clf = 'GaussianProcessClassifier'
		>>>fc.clf_lookup(clf)
			GaussianProcessClassifier(copy_X_train=True, kernel=None,
	             max_iter_predict=100, multi_class='one_vs_rest', n_jobs=1,
	             n_restarts_optimizer=0, optimizer='fmin_l_bfgs_b',
	             random_state=None
			...             
		    n_classes_ : int
		        The number of classes in the training data		    

		>>>fc.clf_obj(clf)
		   GaussianProcessClassifier(copy_X_train=True, kernel=None,
             max_iter_predict=100 ...

	'''
	def __init__(self, clf_type, feature, cv=5, **kwargs):			
		self.X = None
		self.y = None 		  
		self.cv = cv
		self.clf_type = clf_type
		self.feature = feature
		self.kwargs = kwargs
		self._models = dict()
		self._filter()
		self.score_summary_data = None

	def fit(self,X, y):
		'''
		Simply fits X,y and adds to class variable
		
		Params:
			X (array): numpy array
			y (array): numpy array

		'''
		self.X = X
		self.y = y   

	def _grid_search(self, model, params, n_jobs=1, verbose=0, scoring=None, refit=True):
		'''
		Fits the model and params to GridSearchCV

		Params:
			model (sklearn classifier): model to be tested via GridSearchCV
			params (dict): dictionary of parameters to be tested via GridSearchCV

		'''
	    
		print("Running GridSearchCV for %s." % model.__class__.__name__)          
		gs = GridSearchCV(model, params, cv=self.cv, n_jobs=n_jobs, 
		              verbose=verbose, refit=refit)#scoring=scoring
		t0 = time()
		gs.fit(self.X, self.y)
		exec_time = "%0.3fs" % (time() - t0)
		best_est = gs.best_estimator_.__dict__
		best_params = ['{}: {}'.format(key, best_est.get(key)) for key in params.keys()]
		if not best_params:
			best_params = None
		return [gs.best_score_,best_params,exec_time]

	def estimator_selector(self):
		data_hold = []
		for model, params in self._models.items():
			model_name = model.__class__.__name__
			
			scores = self._grid_search(model,params)									
			data_hold.append([model_name]+scores)					
			
		self.score_summary_data = data_hold

	def score_summary(self):

		df1 = pd.DataFrame(self.score_summary_data,
						   columns=['name','score','params','exec_time'])\
									.sort_values('score',ascending=False)		
		return df1 

	def clf_lookup(self,clf):
		print('\r\n'*3)
		p(clf_dict[clf])
		print('\r\n'*3)
		p(dir(clf_dict[clf]))
		print('\r\n'*3)
		p(clf_dict[clf].__dict__)
		print('\r\n'*3)
		print(clf_dict[clf].__doc__)
		
	def clf_obj(self,clf):
		return clf_dict[clf]


