from collections import OrderedDict
import numpy as np
from scipy.stats import pearsonr

class Eval():
	def __init__(self):
		pass

	def assert_inputs_are_numpy_arrays(self, input_list):
		# Assert that inputs are numpy arrays
		for array in input_list:
			assert(type(array) is np.ndarray)

	def evaluate_all(self):
		raise NotImplementedError

class RegEval(Eval):
	'''
	This class evaluates the predicted values of a regression model
	'''
	def __init__(self):
		pass

	def neg_mse(self, y_true, y_pred):
		'''
		Returns the negative of the mean squared error between y_pred and y_true.
		'''
		neg_mse = -1*float((np.sum(np.square(y_pred - y_true)) / len(y_pred)).item())
		return neg_mse

	def pearsonr(self, y_true, y_pred):
		'''
		Returns the Pearson correlation between y_true and y_pred.
		'''
		r, p = pearsonr(y_true, y_pred)
		return r, p

	def evaluate_all(self, y_true, y_pred):
		# Assert inputs are numpy arrays
		self.assert_inputs_are_numpy_arrays([y_true, y_pred])

		# Create metric2score dictionary with regression metrics
		metric2score = OrderedDict()
		metric2score["neg_mse"] = self.neg_mse(y_true, y_pred)
		metric2score["pearsonr"], metric2score["pearsonp"] = self.pearsonr(y_true, y_pred)

		return metric2score, dict()