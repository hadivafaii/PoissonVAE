from utils.plotting import *
from sklearn import metrics as sk_metric
from sklearn import neighbors as sk_neigh
from sklearn import inspection as sk_inspect
from sklearn import linear_model as sk_linear
from sklearn import decomposition as sk_decomp
from sklearn import model_selection as sk_modselect
from statsmodels.stats.multitest import multipletests


def model2key(model_type: str):
	if model_type == 'poisson':
		return 'log_dr'
	elif model_type == 'categorical':
		return 'logits'
	elif model_type in ['gaussian', 'laplace']:
		return 'loc'
	else:
		raise ValueError(model_type)


def model2temp(model_type: str):
	if model_type in ['poisson', 'categorical']:
		return 0.0
	elif model_type in ['gaussian', 'laplace']:
		return 1.0
	else:
		raise ValueError(model_type)


def job_runner_script(
		device: int,
		dataset: str,
		model: str,
		archi: str,
		seed: int = 0,
		args: str = None,
		bash_script: str = 'fit_vae.sh',
		relative_path: str = '.', ):
	s = ' '.join([
		f"'{device}'",
		f"'{dataset}'",
		f"'{model}'",
		f"'{archi}'",
	])
	s = f"{relative_path}/{bash_script} {s}"
	s = f"{s} --seed {seed}"
	if args is not None:
		s = f"{s} {args}"
	return s