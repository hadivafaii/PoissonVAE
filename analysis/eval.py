from .helper import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def knn_analysis(
		x: np.ndarray,
		y: np.ndarray,
		n_iter: int = 20,
		sizes: Iterable[int] = None,
		verbose: bool = False,
		seed: int = 0,
		**kwargs, ):
	knn_defaults = dict(
		n_neighbors=5,
		metric='minkowski',
		weights='uniform',
		algorithm='auto',
		leaf_size=30,
		p=2,
	)
	kwargs = setup_kwargs(knn_defaults, kwargs)

	sizes = sizes if sizes else [
		20, 50, 100, 200, 500,
		1000, 2000, 4000,
	]
	sizes = sorted(filter(
		lambda s: s < len(x),
		sizes
	))
	rng = get_rng(seed)

	range_trn = np.array(range(len(x) // 2))
	vld_ids = np.delete(range(len(x)), range_trn)

	looper = itertools.product(sizes, range(n_iter))
	if len(range_trn) not in sizes:
		looper = list(looper) + [(len(range_trn), 0)]
	pbar = tqdm(
		iterable=looper,
		total=len(looper),
		disable=not verbose,
		ncols=90,
	)
	df = []
	for size, i in pbar:
		pbar.set_description(f"train sample size: {size}, iter #{i}")
		trn_ids = rng.choice(range_trn, size=size, replace=False)
		knn = KNeighborsClassifier(**kwargs).fit(
			x[trn_ids], y[trn_ids])
		report = classification_report(
			y_true=y[vld_ids],
			y_pred=knn.predict(x[vld_ids]),
			output_dict=True,
			zero_division=0,
		)
		df.append({
			'size': [size],
			'iteration': [i],
			'accuracy': [report['accuracy']],
		})
	df = pd.DataFrame(merge_dicts(df))
	df_group = df.select_dtypes(
		include=[np.number]).groupby('size')
	df_summary = pd.DataFrame({
		'mean': df_group.mean()['accuracy'],
		'std': df_group.std()['accuracy']
	})
	return df, df_summary


def sparse_score(z: np.ndarray, cutoff: float = None):
	def _compute_score(axis: int, fix: bool = True):
		m = z.shape[axis]
		numen = np.sum(z, axis=axis) ** 2
		denum = np.sum(z ** 2, axis=axis)
		mask = np.logical_and(  # no spiks
			numen == 0,
			denum == 0,
		)
		denum[denum == 0] = np.finfo('float32').eps
		score = 1 - (numen / denum) / m
		score /= (1 - 1 / m)
		if fix:  # no spiks
			score[mask] = 1.0
		return score

	if isinstance(z, torch.Tensor):
		z = tonp(z)
	if not isinstance(z, np.ndarray):
		z = np.array(z)
	if z.ndim == 1:
		z = z.reshape(-1, 1)
	elif z.ndim == 2:
		pass
	else:
		raise ValueError(f"z.ndim")

	lifetime = _compute_score(0)
	population = _compute_score(1)

	# percentages
	if cutoff is None:
		percents = None
	else:
		counts = collections.Counter(
			np.round(z.ravel()).astype(int))
		portions = {
			k: v / np.prod(z.shape) for
			k, v in counts.most_common()
		}
		try:
			cutoff = next(
				k + 1 for k, v in
				portions.items()
				if v < cutoff
			)
		except StopIteration:
			cutoff = np.inf
		percents = {
			str(k): v for k, v
			in portions.items()
			if k < cutoff
		}
		percents[f'{cutoff}+'] = sum(
			v for k, v in
			portions.items()
			if k >= cutoff
		)
		percents = {
			k: np.round(v * 100, 1) for
			k, v in percents.items()
		}
	return lifetime, population, percents
