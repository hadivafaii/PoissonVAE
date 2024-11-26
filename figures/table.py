from utils.generic import *
from analysis.stats import mu_and_err


def table_entry_shatter(
		dim: int,
		df_agg: pd.DataFrame,
		models: Sequence[str] = None,
		**kwargs, ):

	models = models or [
		'poisson', 'categorical', 'laplace',
		'gaussian', 'gaussian-relu', 'gaussian-exp',
	]
	pre = '\\begin{tabular}{c}'
	post = '\\end{tabular}'

	table = []
	rows = collections.defaultdict(list)
	for mod in models:
		index = (dim, mod)
		try:
			data = np.array(df_agg[index])
			e_str, *_ = mu_and_err(
				dof=4, data=data, **kwargs)
		except KeyError:
			continue
		rows[mod].append(e_str)

	rows = {k: ' & '.join(v) for k, v in rows.items()}
	rows = {k: f"{' ' * 4}{v}" for k, v in rows.items()}
	entry = ' \\\ \n'.join(rows.values())
	entry = f"{pre}\n{entry}\n{post}"
	table.append(entry)

	table = '\n&\n'.join(table)
	return table


def table_entry_knn(
		dim: int,
		df_agg: pd.DataFrame,
		models: Sequence[str] = None,
		sizes: Sequence[str] = None,
		**kwargs, ):

	models = models or [
		'poisson', 'categorical', 'laplace',
		'gaussian', 'gaussian-relu', 'gaussian-exp',
	]
	sizes = sizes or [200, 1000, 5000]
	pre = '\\begin{tabularx}{65mm}{CCC}'
	post = '\\end{tabularx}'

	table = []
	rows = collections.defaultdict(list)
	for mod in models:
		for sz in sizes:
			index = (dim, sz, mod)
			try:
				data = np.array(df_agg[index])
				e_str, *_ = mu_and_err(
					dof=4, data=data, **kwargs)
			except KeyError:
				continue
			rows[mod].append(e_str)

	rows = {k: ' & '.join(v) for k, v in rows.items()}
	rows = {k: f"{' ' * 4}{v}" for k, v in rows.items()}
	entry = ' \\\ \n'.join(rows.values())
	entry = f"{pre}\n{entry}\n{post}"
	table.append(entry)

	table = '\n&\n'.join(table)
	return table


def table_entry_active(
		dataset: str,
		df_agg: pd.DataFrame,
		models: Sequence[str] = None,
		archis: Sequence[str] = None,
		**kwargs, ):

	models = models or [
		'poisson', 'categorical', 'laplace',
		'gaussian', 'gaussian-relu', 'gaussian-exp',
	]
	archis = archis or ['lin', 'conv']
	pre = '\\begin{tabularx}{34mm}{CC}'
	post = '\\end{tabularx}'

	table = []
	rows = collections.defaultdict(list)
	looper = itertools.product(
		models, archis)
	for mod, enc in looper:
		index = (dataset, enc, mod)
		try:
			data = np.array(df_agg[index])
			e_str, *_ = mu_and_err(
				dof=4, data=data, **kwargs)
		except KeyError:
			continue
		rows[mod].append(e_str)

	rows = {k: ' & '.join(v) for k, v in rows.items()}
	rows = {k: f"{' ' * 4}{v}" for k, v in rows.items()}
	entry = ' \\\ \n'.join(rows.values())
	entry = f"{pre}\n{entry}\n{post}"
	table.append(entry)

	table = '\n&\n'.join(table)
	return table


def table_entry_loss(
		model: str,
		df_agg: pd.DataFrame,
		archis: Sequence[str] = None,
		methods: Sequence[str] = None,
		datasets: Sequence[str] = None,
		**kwargs, ):

	archis = archis or ['lin', 'conv']
	methods = methods or ['exact', 'mc', 'st']
	datasets = datasets or ['DOVES', 'CIFAR10-PATCHES', 'MNIST']
	pre = '\\begin{tabularx}{34mm}{CC}'
	post = '\\end{tabularx}'

	table = []
	for ds in datasets:
		rows = collections.defaultdict(list)
		looper = itertools.product(methods, archis)
		for meth, enc in looper:
			index = (model, ds, enc, meth)
			try:
				data = np.array(df_agg[index])
				e_str, *_ = mu_and_err(
					dof=4, data=data, **kwargs)
			except KeyError:
				continue
			rows[meth].append(e_str)

		rows = {k: ' & '.join(v) for k, v in rows.items()}
		rows = {k: f"{' ' * 4}{v}" for k, v in rows.items()}
		entry = ' \\\ \n'.join(rows.values())
		entry = f"{pre}\n{entry}\n{post}"
		table.append(entry)

	table = '\n&\n'.join(table)
	return table
