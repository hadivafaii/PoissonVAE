import re
import os
import json
import h5py
import torch
import shutil
import pickle
import joblib
import random
import inspect
import logging
import pathlib
import argparse
import warnings
import operator
import functools
import itertools
import collections
import numpy as np
import pandas as pd
from torch import nn
from scipy import stats as sp_stats
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as F_vis
from os.path import join as pjoin
from datetime import datetime
from tqdm import tqdm
from typing import *


def true_fn(s: str):  # used as argparse type
	return str(s).lower() == 'true'


def placeholder_fn(val, expected_type):  # used as argparse type
	return val if val == '__placeholder__' else expected_type(val)


def tonp(x: Union[torch.Tensor, np.ndarray]):
	if isinstance(x, np.ndarray):
		return x
	elif isinstance(x, torch.Tensor):
		return x.data.cpu().numpy()
	else:
		raise ValueError(type(x).__name__)


def int_from_str(s: str) -> int:
	matches = re.search(r"\d+", s)
	return int(matches.group())


def alphanum_sort_key(string: str):
	pat = r'^(.*?)([0-9]*\.?[0-9]+)([^0-9]*)$'
	match = re.match(pat, string)
	if match:
		prefix, number, suffix = match.groups()
		return prefix, float(number), suffix
	return string, 0, ''


def flatten_np(
		x: np.ndarray,
		start_dim: int = 0,
		end_dim: int = -1, ):
	shape = x.shape
	if start_dim < 0:
		start_dim += len(shape)
	if end_dim < 0:
		end_dim += len(shape)
	prefix = shape[:start_dim]
	suffix = shape[end_dim+1:]
	middle = np.prod(shape[start_dim:end_dim+1])
	shape = (*prefix, middle, *suffix)
	return x.reshape(shape)


def cat_map(x: list, axis: int = 0):
	out = []
	for a in x:
		if len(a):
			out.append(np.concatenate(
				a, axis=axis))
		else:
			out.append(a)
	return out


def get_tval(
		dof: int,
		ci: float = 0.95,
		two_sided: bool = True, ):
	if two_sided:
		ci = (1 + ci) / 2
	return sp_stats.t.ppf(ci, dof)


def make_logger(
		name: str,
		path: str,
		level: int,
		module: str = None, ) -> logging.Logger:
	os.makedirs(path, exist_ok=True)
	logger = logging.getLogger(module)
	logger.setLevel(level)
	file = pjoin(path, f"{name}.log")
	file_handler = logging.FileHandler(file)
	formatter = logging.Formatter(
		'%(asctime)s : %(levelname)s : %(name)s : %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def get_rng(
		x: Union[int, np.random.Generator, random.Random] = 42,
		use_np: bool = True, ):
	if isinstance(x, int):
		if use_np:
			return np.random.default_rng(seed=x)
		else:
			return random.Random(x)
	elif isinstance(x, (np.random.Generator, random.Random)):
		return x
	else:
		print('Warning, invalid random state. returning default')
		return np.random.default_rng(seed=42)


def divide_list(lst: list, n: int):
	k, m = divmod(len(lst), n)
	lst_divided = [
		lst[
			i * k + min(i, m):
			(i + 1) * k + min(i + 1, m)
		] for i in range(n)
	]
	return lst_divided


def add_home(path: str):
	if '/home/' not in path:
		return pjoin(os.environ['HOME'], path)
	return path


def setup_kwargs(defaults, kwargs):
	if not kwargs:
		return defaults
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	return kwargs


def get_default_params(f: Callable):
	params = inspect.signature(f).parameters
	return {
		k: p.default for
		k, p in params.items()
	}


def get_all_init_params(cls: Callable):
	init_params = {}
	classes_to_process = [cls]

	while classes_to_process:
		current_cls = classes_to_process.pop(0)
		for base_cls in current_cls.__bases__:
			if base_cls != object:
				classes_to_process.append(base_cls)
		sig = inspect.signature(current_cls.__init__)
		init_params.update(sig.parameters)

	return init_params


def filter_kwargs(
		fn: Callable,
		kw: dict = None, ):
	if not kw:
		return {}
	try:
		if isinstance(fn, type):  # class
			params = get_all_init_params(fn)
		elif callable(fn):  # function
			params = inspect.signature(fn).parameters
		else:
			raise ValueError(type(fn).__name__)
		return {
			k: v for k, v
			in kw.items()
			if k in params
		}
	except ValueError:
		return kw


def obj_attrs(
		obj: object,
		with_base: bool = True, ):
	# get params
	sig = inspect.signature(obj.__init__)
	params = dict(sig.parameters)
	if with_base:
		params.update(get_all_init_params(type(obj)))
	# get rid of self, args, kwargs
	vals = {
		k: getattr(obj, k) for
		k, p in params.items()
		if _param_checker(k, p, obj)
	}
	# get rid of functions
	vals = {
		k: v for k, v in vals.items()
		if not isinstance(v, Callable)
	}
	# remove directories
	vals = {
		k: v for k, v in vals.items()
		if '_dir' not in k
	}
	return vals


def save_obj(
		obj: Any,
		file_name: str,
		save_dir: str,
		mode: str = None,
		verbose: bool = True, ):
	_allowed_modes = [
		'npy', 'df',
		'pkl', 'joblib',
		'html', 'json', 'txt',
	]
	_ext = file_name.split('.')[-1]
	if _ext in _allowed_modes:
		mode = _ext
	else:
		if mode is None:
			msg = 'invalid file extension: '
			msg += f"{_ext}, mode: {mode}"
			raise RuntimeError(msg)
		else:
			file_name = f"{file_name}.{mode}"
	assert mode in _allowed_modes, \
		f"available modes:\n{_allowed_modes}"

	path = pjoin(save_dir, file_name)
	op_mode = 'w' if mode in ['html', 'json', 'txt'] else 'wb'
	with open(path, op_mode) as f:
		if mode == 'npy':
			np.save(f.name, obj)
		elif mode == 'df':
			pd.to_pickle(obj, f.name)
		elif mode == 'pkl':
			# noinspection PyTypeChecker
			pickle.dump(obj, f)
		elif mode == 'joblib':
			joblib.dump(obj, f)
		elif mode == 'html':
			f.write(obj)
		elif mode == 'json':
			json.dump(obj, f, indent=4)
		elif mode == 'txt':
			for line in obj:
				f.write(line)
		else:
			raise RuntimeError(mode)
	if verbose:
		print(f"[PROGRESS] '{file_name}' saved at\n{save_dir}")
		return None
	return path


def merge_dicts(
		dict_list: List[dict],
		verbose: bool = False, ) -> Dict[str, list]:
	merged = collections.defaultdict(list)
	dict_items = map(operator.methodcaller('items'), dict_list)
	iterable = itertools.chain.from_iterable(dict_items)
	kws = {
		'leave': False,
		'disable': not verbose,
		'desc': "...merging dicts",
	}
	for k, v in tqdm(iterable, **kws):
		merged[k].extend(v)
	return dict(merged)


def expected_relu(loc: np.ndarray, scale: np.ndarray):
	"""
	E[ReLU(z)] = scale * phi(loc/scale) + loc * Phi(loc/scale)
	"""
	loc = np.asarray(loc)
	scale = np.asarray(scale)

	# Handle scale = 0 to avoid division by zero
	safe_scale = np.where(scale > 0, scale, 1.0)
	alpha = loc / safe_scale

	x = (
		scale * sp_stats.norm.pdf(alpha) +
		loc * sp_stats.norm.cdf(alpha)
	)
	y = np.maximum(loc, 0)  # scale = 0 → delta → just ReLU(loc)
	result = np.where(scale > 0, x, y)

	return result


def find_last_contiguous_zeros(mask: np.ndarray, w: int):
	# mask = hist > 0.0
	m = mask.astype(bool)
	zero_count = 0
	for idx, val in enumerate(m[::-1]):
		if val == 0:
			zero_count += 1
		else:
			zero_count = 0

		if zero_count == w:
			return len(m) - (idx - w + 2)
	return 0


def find_critical_ids(mask: np.ndarray):
	# mask = hist > 0.0
	m = mask.astype(bool)

	first_zero = 0
	for i in range(1, len(m)):
		if m[i-1] and not m[i]:
			first_zero = i
			break

	last_zero = -1
	for i in range(len(m) - 2, -1, -1):
		if not m[i] and m[i+1]:
			last_zero = i
			break

	return first_zero, last_zero


def time_dff_string(start: str, stop: str):
	d, h, m, _ = time_difference(start, stop)
	delta_t = f"{h}h, {m}m"
	if d > 0:
		delta_t = f"{d}d, {delta_t}"
	return delta_t


def time_difference(
		start: str,
		stop: str,
		fmt: str = '%Y_%m_%d,%H:%M', ):
	start_datetime = datetime.strptime(start, fmt)
	stop_datetime = datetime.strptime(stop, fmt)
	diff = stop_datetime - start_datetime

	hours, remainder = divmod(diff.seconds, 3600)
	mins, seconds = divmod(remainder, 60)

	return diff.days, hours, mins, seconds


def now(include_hour_min: bool = False):
	fmt = "%Y_%m_%d"
	if include_hour_min:
		fmt += ",%H:%M"
	return datetime.now().strftime(fmt)


def _param_checker(k, p, obj):
	# 2nd cond gets rid of args, kwargs
	return k != 'self' and int(p.kind) == 1 and hasattr(obj, k)
