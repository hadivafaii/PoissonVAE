from utils.generic import *
from sklearn import linear_model
from sklearn.metrics import r2_score, classification_report
from scipy.spatial import distance as sp_dist


def clf_score(z_dict: dict, g_dict: dict):
	# fit logistic regression clf
	lr = linear_model.LogisticRegression().fit(
		X=z_dict['trn'], y=g_dict['trn'])
	# compute accuracies
	accu = {}
	for k, z in z_dict.items():
		accu[k] = classification_report(
			y_true=g_dict[k],
			y_pred=lr.predict(z),
			output_dict=True,
		)['accuracy']
	return accu


def untangle_score(z_dict: dict, g_dict: dict):
	# fit linear regression
	lr = linear_model.LinearRegression().fit(
		X=z_dict['trn'], y=g_dict['trn'])
	# compute scores
	r2, corr = {}, {}
	for k, z in z_dict.items():
		# get true & pred
		true = g_dict[k]
		pred = lr.predict(z)
		# compute scores
		r2[k] = r2_score(
			y_true=true, y_pred=pred,
			multioutput='raw_values',
		)
		corr[k] = 1 - np.diag(sp_dist.cdist(
			XA=true.T, XB=pred.T,
			metric='correlation',
		))

	output = {
		'r2_zg': dict(r2),
		'corr_zg': dict(corr),
	}
	return output
