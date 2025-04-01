from utils.generic import *
from base.utils_model import load_quick, load_model_lite, compute_r2
from analysis.linear import clf_score, untangle_score
from analysis.clf import clf_analysis
from analysis.eval import (
	sparse_score,
	knn_analysis,
	model2temp,
	model2key,
)
import gc


# Copied from _IterativeVAE
def perform_analysis(
		results_dir: str,
		device: str | torch.device,
		attrs: Sequence[str] = None,
		attrs_tr: Sequence[str] = None,
		root: str = 'Dropbox/chkpts/PoissonVAE',
		override_fits: bool = False,
		verbose: bool = True, ):

	attrs = attrs or [
		'seed', 'dataset', 'type',
		'latent_act', 'n_latents',
	]
	attrs_tr = attrs_tr or ['kl_beta']

	root = add_home(root)
	fits = sorted(
		os.listdir(root),
		key=alphanum_sort_key,
	)
	# the big for loop
	for name in tqdm(fits, position=0):
		save_name = f"{name}.npy"
		is_already_fit = os.path.isfile(pjoin(
			results_dir, save_name))
		if is_already_fit and not override_fits:
			continue

		# (1) load trainer
		try:
			tr, meta = load_model_lite(
				pjoin(root, name),
				device=device,
				shuffle=False,
				verbose=False,
				strict=True,
			)
		except StopIteration:
			print(f"missing: {name}")
			continue

		# (2) info
		info = {k: meta[k] for k in ['checkpoint', 'timestamp']}
		info.update({a: getattr(tr.model.cfg, a, None) for a in attrs})
		info['str_model'] = f"{tr.model.cfg.model_str}+amort"
		info['archi'] = tr.model.cfg.attr2archi()
		info['seq_len'] = 1
		info.update({a: getattr(tr.cfg, a, None) for a in attrs_tr})
		info['n_params'] = sum([p.numel() for p in tr.parameters()])
		info['n_iters_train'] = tr.n_iters

		# (3) compute loss, xtract ftrs
		data, loss, etc = tr.validate(full_data=True)
		loss['mse_map'] = tr.mse_map()
		results = {k: v.mean() for k, v in loss.items()}
		lifetime, population, _ = sparse_score(data['z'])
		results['lifetime'] = lifetime.mean()
		results['population'] = population.mean()
		results['%-zeros'] = np.mean(data['z'] == 0)
		results['r2'] = compute_r2(
			true=tr.to(data['x']).flatten(start_dim=1),
			pred=tr.to(data['y']).flatten(start_dim=1),
		).mean().item()
		results['samples_final'] = data['z']

		# (4) clf/untangle score?
		repres_key = model2key(tr.model.cfg.type)
		if tr.model.cfg.dataset.endswith('MNIST'):
			data_trn, _, etc_trn = tr.forward(
				'trn', full_data=True)
			z_dict = {
				'trn': etc_trn[repres_key],
				'vld': etc[repres_key],
			}
			g_dict = {
				'trn': data_trn['g'],
				'vld': data['g'],
			}
			results['clf_accuracy'] = clf_score(
				z_dict=z_dict, g_dict=g_dict)

		elif tr.model.cfg.dataset.startswith('BALLS'):
			data_trn, _, etc_trn = tr.forward(
				'trn', full_data=True)
			data_tst, _, etc_tst = tr.forward(
				'tst', full_data=True)
			z_dict = {
				'trn': etc_trn[repres_key],
				'vld': etc[repres_key],
				'tst': etc_tst[repres_key],
			}
			g_dict = {
				'trn': data_trn['g'][:, [1, 3]],
				'vld': data['g'][:, [1, 3]],
				'tst': data_tst['g'][:, [1, 3]],
			}
			untangle = untangle_score(
				z_dict=z_dict, g_dict=g_dict)
			results = {**results, **untangle}

		# (4) save
		save_obj(
			obj={'info': info, 'results': results},
			save_dir=results_dir,
			file_name=save_name,
			verbose=verbose,
		)

		# (5) clean up
		del tr
		torch.cuda.empty_cache()
		gc.collect()

	return


def analyze_fits(
		fits: Sequence[str],
		device: torch.device,
		analysis_mode: str,
		knn_n_iter: int = 100,
		clf_type: str = 'logreg',
		sparse_key: str = 'lifetime',
		attrs: Sequence[str] = None,
		attrs_tr: Sequence[str] = None,
		root: str = 'Dropbox/chkpts/PoissonVAE', ):
	attrs = attrs or [
		'dataset', 'type',
		'enc_type', 'dec_type', 'latent_act',
		'n_latents', 'n_categories', 'seed',
	]
	attrs_tr = attrs_tr or ['kl_beta']
	df = collections.defaultdict(list)
	for name in tqdm(fits):
		# load trainer
		tr, meta = load_quick(
			pjoin(add_home(root), name),
			device=device,
			verbose=False,
			lite=True,
		)
		# extract info
		vals = {k: meta[k] for k in ['checkpoint', 'timestamp']}
		vals.update({a: getattr(tr.model.cfg, a, None) for a in attrs})
		vals.update({a: getattr(tr.cfg, a, None) for a in attrs_tr})
		vals['n_params'] = sum([p.numel() for p in tr.parameters()])
		vals['method'] = tr.cfg.method
		# compute loss, xtract ftrs
		data, loss, etc = tr.validate()
		loss['mse_map'] = tr.mse_map()

		if analysis_mode == 'main':
			# add portion active neurons
			dead = tr.find_dead_neurons(kl=loss['kl_diag'])
			vals['active'] = (~dead).sum() / len(dead)
			# add loss
			loss_avg = {k: v.mean() for k, v in loss.items()}
			loss_avg['nelbo'] = loss_avg['mse'] + loss_avg['kl']
			vals.update(loss_avg)
			# add temperature realted stuff
			vals['temp_anneal'] = tr.cfg.temp_anneal_type
			vals['temp_start'] = tr.cfg.temp_start
			vals['temp_stop'] = tr.cfg.temp_stop
			vals['hard_fwd'] = str(tr.model.cfg.hard_fwd)
			# turn vals into list
			vals = {
				k: [v] for k, v
				in vals.items()
			}

		elif analysis_mode == 'sparse':
			# add sparse score / portion dead neurons
			lifetime, population, _ = sparse_score(
				z=np.abs(data['z']), cutoff=None)
			if sparse_key == 'lifetime':
				sprs_measure = lifetime
			elif sparse_key == 'population':
				sprs_measure = population
			else:
				raise ValueError(sparse_key)
			vals = {
				k: [v] * len(sprs_measure)
				for k, v in vals.items()
			}
			vals[sparse_key] = sprs_measure
			vals['neuron_i'] = list(range(len(
				sprs_measure)))

		elif analysis_mode == 'knn':
			key = model2key(tr.model.cfg.type)
			df_knn, _ = knn_analysis(
				x=flatten_np(etc.get(key), start_dim=1),
				y=tonp(tr.dl_vld.dataset.tensors[1]).astype(int),
				seed=tr.model.cfg.seed,
				sizes=[200, 1000],
				n_iter=knn_n_iter,
			)
			vals = {
				k: [v] * len(df_knn)
				for k, v in vals.items()
			}
			vals.update(df_knn.to_dict(
				orient='list'))

		elif analysis_mode in ['clf', 'shatter']:
			t = model2temp(tr.model.cfg.type)
			key = model2key(tr.model.cfg.type)
			# fwd trn
			data, etc = {}, {}
			for item in ['trn', 'vld']:
				data[item], _, etc[item] = tr.forward(
					dl_name=item, temp=t, full_data=True)
			ftrs = {
				k: flatten_np(v[key], start_dim=1)
				for k, v in etc.items()
			}
			# get disjoint groups
			labels = {
				k: v['g'] for k, v
				in data.items()
			}
			df_clf = clf_analysis(
				x=ftrs,
				labels=labels,
				clf_type=clf_type,
				mode=analysis_mode,
			)
			vals = {
				k: [v] * len(df_clf)
				for k, v in vals.items()
			}
			vals.update(df_clf.to_dict(
				orient='list'))

		else:
			raise ValueError(analysis_mode)

		# extend values
		for k, v in vals.items():
			df[k].extend(v)
	df = pd.DataFrame(df)
	df = _fuse(df)
	return df


def add_nelbo_diff(df: pd.DataFrame):
	df = df.drop(columns=[
		'checkpoint', 'timestamp', 'n_params', 'dec_type',
		'mse', 'mse_map', 'kl', 'kl_diag', 'kl_beta',
	])
	df = df.loc[df['type'].isin(['poisson', 'gaussian'])]
	df['nelbo_diff'] = np.nan

	datasets = ['DOVES', 'CIFAR10-PATCHES', 'MNIST']
	models = ['poisson', 'gaussian']
	encoders = ['lin', 'conv']
	looper = itertools.product(
		datasets, models, encoders)
	for ds, mod, enc in looper:
		cond = (
			(df['type'] == mod) &
			(df['dataset'] == ds) &
			(df['enc_type'] == enc)
		)
		_df = df.loc[cond].copy()
		best = _df['nelbo'].min()
		diff = 100 * (_df['nelbo'] - best) / best
		df.loc[cond, 'nelbo_diff'] = diff.values
	return df


def sort_fits(root: str = 'Dropbox/chkpts/PoissonVAE'):
	fits = sorted(os.listdir(add_home(root)))
	fits_st = [
		f for f in fits if
		'st_chewie' in f
	]
	fits_all = [
		f for f in fits if
		'st_chewie' not in f
	]

	fn = ['-relu', '-softplus', '-exp', '-square']
	fits = [
		f for f in fits_all if not
		('-b' in f or any(e in f for e in fn))
	]
	fits_etc = [f for f in fits_all if f not in fits]

	assert len(set(fits_all)) == \
		len(set(fits)) + len(set(fits_etc))

	fits, fits_st, fits_etc = map(
		sorted, [fits, fits_st, fits_etc]
	)
	return fits, fits_st, fits_etc


def _fuse(df: pd.DataFrame):
	# add n_dims
	df1 = df.copy()
	n_dims = np.nanprod(np.concatenate([
		np.reshape(df['n_latents'], (1, -1)).astype(float),
		np.reshape(df['n_categories'], (1, -1)).astype(float),
	]), axis=0).astype(int)
	df1.insert(7, 'n_dims', n_dims)
	df1.drop(
		columns=[
			'n_latents',
			'n_categories'],
		inplace=True,
	)
	# fuse latent_act
	df2 = df1.copy()
	df2['latent_act'] = df2['latent_act'].fillna('none')
	df2['latent_act'] = '-' + df2['latent_act']
	df2.loc[df2['latent_act'] == '-none', 'latent_act'] = ''
	df2['type'] += df2['latent_act']
	df2.drop(
		columns='latent_act',
		inplace=True,
	)
	# fuse kl_beta
	df3 = df2.copy()
	df3['kl_beta_str'] = df3['kl_beta'].apply(lambda x: f"{x:0.2g}")
	df3['kl_beta_str'] = '-b' + df3['kl_beta_str']
	df3.loc[df3['kl_beta_str'] == '-b1', 'kl_beta_str'] = ''
	df3['type'] += df3['kl_beta_str']
	df3.drop(
		columns='kl_beta_str',
		inplace=True,
	)
	return df3
