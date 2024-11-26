from figures.fighelper import *


def nelbo_point(
		results: pd.DataFrame,
		add_group: bool = False,
		legend: bool = False,
		display: bool = True,
		remove_lbls: bool = True,
		archis: Sequence[str] = None,
		datasets: Sequence[str] = None,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'scale': 1.5,
		'point_scale': 1.0,
		'strip_alpha': 0.7,
		'strip_size': 10,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	archis = archis or ['lin', 'conv']
	datasets = datasets or ['DOVES', 'CIFAR10-PATCHES', 'MNIST']
	nrows, ncols = len(archis), len(datasets)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=(
			ncols * kwargs['scale'],
			nrows * kwargs['scale']),
		dpi=kwargs['dpi'],
		sharey='all',
		sharex='all',
	)
	for idx, ax in enumerate(axes.flat):
		i, j = idx // ncols, idx % ncols
		_df = results.loc[
			(results['enc_type'] == archis[i]) &
			(results['dataset'] == datasets[j])
		].copy()
		sns.stripplot(
			data=_df,
			y='nelbo_diff',
			x='method',
			hue='type',
			order=['exact', 'mc', 'st'],
			hue_order=['poisson', 'gaussian'],
			palette=get_palette()[0],
			marker='$\circ$',
			legend=False,
			jitter=True,
			dodge=False,
			alpha=kwargs['strip_alpha'],
			size=kwargs['strip_size'],
			ax=ax,
		)
		if add_group:
			sns.pointplot(
				data=_df,
				y='nelbo_diff',
				x='method',
				hue='type',
				order=['exact', 'mc', 'st'],
				hue_order=['poisson', 'gaussian'],
				palette='muted',
				linestyles='',
				markers='x',
				errwidth=0,
				capsize=None,
				dodge=True,
				scale=kwargs['point_scale'],
				ax=ax,
			)
		if remove_lbls:
			ax.set(
				xlabel='',
				ylabel='',
				xticklabels=[],
			)
		else:
			tit = f"{archis[i]}\n{datasets[j]}"
			ax.set_title(label=tit, fontsize=9)
		ax.grid()
		if not legend:
			move_legend(ax)

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes
