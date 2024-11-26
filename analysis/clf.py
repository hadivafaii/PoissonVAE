from utils.generic import *
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def clf_analysis(
		mode: str,
		x: Dict[str, np.ndarray],
		labels: Dict[str, np.ndarray],
		clf_type: str = 'logreg',
		verbose: bool = False,
		**kwargs, ):
	def _get_clf(**kws):
		if clf_type == 'logreg':
			return LogisticRegression(**kws)
		elif clf_type == 'svm':
			return LinearSVC(dual='auto', **kws)
		elif clf_type == 'knn':
			return KNeighborsClassifier(**kws)
		else:
			raise NotImplementedError(clf_type)

	df = []
	if mode == 'clf':
		clf = _get_clf(**kwargs)
		clf.fit(x['trn'], labels['trn'])
		pred = clf.predict(x['vld'])
		report = classification_report(
			y_true=labels['vld'],
			y_pred=pred,
			output_dict=True,
		)
		df.append({
			'classifier': [type(clf).__name__],
			'accuracy': [report['accuracy']],
		})
	elif mode == 'shatter':
		digits = list(labels.values())[0].astype(int)
		groups = disjoint_groups(sorted(np.unique(digits)))
		for idx, (c0, c1) in tqdm(
				enumerate(groups),
				disable=not verbose,
				total=len(groups),
				desc=clf_type,
				ncols=80):
			y = digit2category(labels['trn'], c1)
			y_vld = digit2category(labels['vld'], c1)

			clf = _get_clf(**kwargs)
			clf.fit(x['trn'], y)
			pred = clf.predict(x['vld'])

			report = classification_report(
				y_true=y_vld,
				y_pred=pred,
				output_dict=True,
			)
			df.append({
				'group_idx': [idx],
				'category_0': [c0],
				'category_1': [c1],
				'classifier': [type(clf).__name__],
				'accuracy': [report['accuracy']],
			})
	else:
		raise NotImplementedError(mode)

	return pd.DataFrame(merge_dicts(df))


def disjoint_groups(data: Sequence):
	assert len(data) % 2 == 0, "# elements must be even"
	all_combos = list(itertools.combinations(
		iterable=data, r=len(data) // 2))

	result = []
	for combo in all_combos:
		complement = tuple(set(data) - set(combo))
		result.append((list(combo), list(complement)))

	return result


def digit2category(
		digits: np.ndarray,
		category: list, ):
	return np.isin(
		element=digits.astype(int),
		test_elements=category,
	).astype(int)
