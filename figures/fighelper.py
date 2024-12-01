from utils.plotting import *


def get_betas(df: pd.DataFrame):
	betas: List[Union[float, str]] = ['ae']
	betas += sorted([
		b for b in df['beta'].unique()
		if isinstance(b, float)
	])
	return betas


def get_palette():
	# poisson
	betas = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 4.0]
	colors = sns.color_palette(
		'crest', n_colors=len(betas))
	palette = {
		'poisson' if b == 1.0 else f"poisson-{b:0.2g}":
			c for b, c in zip(betas, colors)
	}
	palette_beta = {
		b: c for b, c in zip(betas, colors)
	}
	# gaussian
	items = ['-relu', '', '-exp']
	colors = sns.color_palette(
		'flare', n_colors=len(items))
	palette.update({
		'gaussian' + k: c for k, c
		in zip(items, colors)
	})
	# laplace, categorical
	muted = sns.color_palette('muted')
	palette.update({
		'laplace': muted[2],
		'categorical': muted[5],
	})
	# lca, ista
	palette.update({
		'lca': '#6f6f6f',
		'ista': '#aeaeae',
	})
	# lambda (lca)
	lamb = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.7, 1.0]
	colors = sns.cubehelix_palette(n_colors=len(lamb))
	palette_lamb = {
		lamb: c for lamb, c in zip(lamb, colors)
	}
	return palette, palette_beta, palette_lamb


def show_palette(palette: dict, ncols: int = 8):
	nrows = -(-len(palette) // ncols)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=(ncols * 1.2, nrows * 1.4),
	)
	axes = axes.flatten()

	for i, (key, color) in enumerate(palette.items()):
		axes[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
		axes[i].axis('off')
		axes[i].set_title(key)

	for j in range(len(palette), len(axes)):
		axes[j].axis('off')

	plt.show()
	return


def plot_bar(df: pd.DataFrame, display: bool = True, **kwargs):
	defaults = dict(
		x='x',
		y='y',
		figsize_y=7,
		figsize_x=0.7,
		tick_labelsize_x=15,
		tick_labelsize_y=15,
		ylabel_fontsize=20,
		title_fontsize=18,
		vals_fontsize=13,
		title_y=1,
	)
	kwargs = setup_kwargs(defaults, kwargs)
	figsize = (
		kwargs['figsize_x'] * len(df),
		kwargs['figsize_y'],
	)
	fig, ax = create_figure(1, 1, figsize)
	bp = sns.barplot(data=df, x=kwargs['x'], y=kwargs['y'], ax=ax)
	barplot_add_vals(bp, fontsize=kwargs['vals_fontsize'])
	ax.tick_params(
		axis='x',
		rotation=-90,
		labelsize=kwargs['tick_labelsize_x'],
	)
	ax.tick_params(
		axis='y',
		labelsize=kwargs['tick_labelsize_y'],
	)
	val = np.nanmean(df[kwargs['y']]) * 100
	title = r'avg $R^2 = $' + f"{val:0.1f} %"
	ax.set_title(
		label=title,
		y=kwargs['title_y'],
		fontsize=kwargs['title_fontsize'],
	)
	ax.set_ylabel(
		ylabel=r'$R^2$',
		fontsize=kwargs['ylabel_fontsize'],
	)
	ax.set(xlabel='', ylim=(0, 1))
	ax.grid()
	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


CAT2TEX = {
	'fixate0': '\\fixate{0}',
	'fixate1': '\\fixate{1}',
	'obj1': '\\obj{1}',
}

LBL2TEX = {
	'fix_x': r'$F_x$',
	'fix_y': r'$F_y$',
	'slf_v_x': r'$V_{self, x}$',
	'slf_v_y': r'$V_{self, y}$',
	'slf_v_z': r'$V_{self, z}$',
	'obj0_x': r'$X_{obj}$',
	'obj0_y': r'$Y_{obj}$',
	'obj0_z': r'$Z_{obj}$',
	'obj0_v_x': r'$V_{obj, x}$',
	'obj0_v_y': r'$V_{obj, y}$',
	'obj0_v_z': r'$V_{obj, z}$',
}
