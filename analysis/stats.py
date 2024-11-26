from utils.generic import *
from statsmodels.stats.multitest import multipletests


def ttest(
		value: str,
		df: pd.DataFrame,
		by: Sequence[str],
		test_conds: List[Tuple[str, str]] = None,
		method: str = 'fdr_bh',
		alpha: float = 0.01, ):

	df_agg = df.groupby(by=by)[value].agg(list)
	df_ttest = collections.defaultdict(list)

	test_conds = test_conds or list(itertools.combinations(
		df[by[-1]].unique(), r=2))
	looper_keys = by[:-1]
	looper = itertools.product(*tuple(
		df[k].unique().tolist()
		for k in looper_keys
	))

	for items in looper:
		for c1, c2 in test_conds:
			vals = {
				k: v for k, v in
				zip(looper_keys, items)
			}
			try:
				x1 = np.array(df_agg[(*items, c1)])
				x2 = np.array(df_agg[(*items, c2)])
			except KeyError:
				continue
			good = np.logical_and(
				np.isfinite(x1),
				np.isfinite(x2),
			)
			test = sp_stats.ttest_rel(
				x1[good], x2[good])
			vals.update({
				'cond1': c1,
				'cond2': c2,
				't': test.statistic,
				'pvals': test.pvalue,
			})
			for k, v in vals.items():
				df_ttest[k].append(v)

	df_ttest = pd.DataFrame(df_ttest)

	rejected_corrected, pvals_corrected, *_ = multipletests(
		df_ttest['pvals'], alpha=alpha, method=method)
	df_ttest[f'pvals_{method}'] = pvals_corrected
	df_ttest['reject'] = rejected_corrected

	return df_agg, df_ttest


def mu_and_err(
		dof: int,
		data: np.ndarray,
		fmt: str = '0.1f',
		n_resamples: int = int(1e6),
		ci: float = 0.99, ):
	data = data[np.isfinite(data)]
	se = sp_stats.bootstrap(
		data=(data,),
		n_resamples=n_resamples,
		statistic=np.nanmean,
		method='BCa',
	).standard_error

	mu = np.nanmean(data)
	err = se * get_tval(
		dof=dof, ci=ci)

	# make table entry
	mu_str = f"{'{'}{mu:{fmt}}{'}'}"
	err_str = f"{err:{fmt}}"
	mu_str, err_str = map(
		_strip0, [mu_str, err_str])
	err_str = f"{'{'}{err_str}{'}'}"
	entry = f"\entry{mu_str}{err_str}"

	return entry, mu, err


def _strip0(s: str):
	if s.startswith('0'):
		return s.lstrip('0')
	return s
