from utils.plotting import *
from main.config_vae import *
from torch import distributions as dists


def temp_anneal_linear(
		n_iters: int,
		t0: float = 1.0,
		t1: float = 0.1,
		portion: float = 0.7, ):
	temperatures = np.ones(n_iters) * t1
	i = int(np.ceil(portion * n_iters))
	temperatures[:i] = np.linspace(t0, t1, i)
	return temperatures


def temp_anneal_exp(
		n_iters: int,
		t0: float = 1.0,
		t1: float = 0.1,
		portion: float = 0.7,
		rate: float = 'infer', ):
	n = int(np.ceil(n_iters * portion))
	n = min(n, n_iters)
	if rate == 'infer':
		rate = - (n/(n-1)) * np.log(t1/100)
	else:
		assert isinstance(rate, float)
	temperatures = np.ones(n_iters) * t1
	for i in range(n):
		coef = np.exp(-rate * i / n)
		t = t1 + (t0 - t1) * coef
		temperatures[i] = t
	return temperatures


def beta_anneal_cosine(
		n_iters: int,
		start: float = 0.0,
		stop: float = 1.0,
		n_cycles: int = 4,
		portion: float = 0.5,
		beta: float = 1.0, ):
	period = n_iters / n_cycles
	step = (stop-start) / (period*portion)
	betas = np.ones(n_iters) * beta
	for c in range(n_cycles):
		v, i = start, 0
		while v <= stop:
			val = (1 - np.cos(v*np.pi)) * beta / 2
			betas[int(i+c*period)] = val
			v += step
			i += 1
	return betas


def beta_anneal_linear(
		n_iters: int,
		beta: float = 1,
		anneal_portion: float = 0.3,
		constant_portion: float = 0,
		min_beta: float = 1e-4, ):
	betas = np.ones(n_iters) * beta
	a = int(np.ceil(constant_portion * n_iters))
	b = int(np.ceil((constant_portion + anneal_portion) * n_iters))
	betas[:a] = min_beta
	betas[a:b] = np.linspace(min_beta, beta, b - a)
	return betas


class AverageMeter(object):
	def __init__(self):
		self.val = 0
		self.sum = 0
		self.avg = 0
		self.cnt = 0

	def reset(self):
		self.val = 0
		self.sum = 0
		self.avg = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def load_quick(s: str, lite: bool, **kwargs):
	if lite:
		tr, meta = load_model_lite(s, **kwargs)
	else:
		tr, meta = load_model(*s.split('/'), **kwargs)
	return tr, meta


def load_model_lite(
		path: str,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		**kwargs, ):
	# load model
	cfg = next(
		e for e in os.listdir(path)
		if e.startswith('Config')
		and e.endswith('.json')
		and 'Train' not in e
	)
	fname = cfg.split('.')[0]
	cfg = pjoin(path, cfg)
	with open(cfg, 'r') as f:
		cfg = json.load(f)
	# extract key
	key = next(
		k for k, cls in
		CFG_CLASSES.items()
		if fname == cls.__name__
	)
	# load cfg/model
	cfg = CFG_CLASSES[key](**cfg)
	from main.vae import MODEL_CLASSES
	model = MODEL_CLASSES[key](
		cfg, verbose=verbose)

	# load state dict
	fname_pt = next(
		f for f in os.listdir(path)
		if f.split('.')[-1] == 'pt'
	)
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(state_dict, 'cpu')
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)

	# set chkpt_dir & timestamp
	model.chkpt_dir = path
	timestamp = state_dict['metadata'].get('timestamp')
	if timestamp is not None:
		model.timestamp = timestamp

	# load trainer
	cfg_train = next(
		e for e in os.listdir(path)
		if e.startswith('Config')
		and e.endswith('.json')
		and 'Train' in e
	)
	fname = cfg_train.split('.')[0]
	cfg_train = pjoin(path, cfg_train)
	with open(cfg_train, 'r') as f:
		cfg_train = json.load(f)
	if fname.endswith('VAE'):
		from main.train_vae import TrainerVAE
		cfg_train = ConfigTrainVAE(**cfg_train)
		trainer = TrainerVAE(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			**kwargs,
		)
	else:
		raise NotImplementedError

	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
		if timestamp is not None:
			trainer.model_ema.timestamp = timestamp

	# optim, etc.
	if strict:
		trainer.optim.load_state_dict(
			state_dict['optim'])
		trainer.scaler.load_state_dict(
			state_dict['scaler'])
		if trainer.optim_schedule is not None:
			trainer.optim_schedule.load_state_dict(
				state_dict.get('scheduler', {}))
	stats = state_dict['metadata'].pop('stats', {})
	trainer.stats.update(stats)
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
	}
	return trainer, metadata


def load_model(
		model_name: str,
		fit_name: Union[str, int] = -1,
		checkpoint: int = -1,
		device: str = 'cpu',
		strict: bool = True,
		verbose: bool = False,
		path: str = 'Projects/PoissonVAE/models',
		**kwargs, ):
	# cfg model
	path = pjoin(add_home(path), model_name)
	fname = next(s for s in os.listdir(path) if 'json' in s)
	with open(pjoin(path, fname), 'r') as f:
		cfg = json.load(f)
	# extract key
	fname = fname.split('.')[0]
	key = next(
		k for k, cls in
		CFG_CLASSES.items()
		if fname == cls.__name__
	)
	# load cfg/model
	cfg = CFG_CLASSES[key](**cfg)
	from main.vae import MODEL_CLASSES
	model = MODEL_CLASSES[key](
		cfg, verbose=verbose)

	# now enter the fit folder
	if isinstance(fit_name, str):
		path = pjoin(path, fit_name)
	elif isinstance(fit_name, int):
		path = sorted(filter(
			os.path.isdir, [
				pjoin(path, e) for e
				in os.listdir(path)
			]
		), key=_sort_fn)[fit_name]
	else:
		raise ValueError(fit_name)
	files = sorted(os.listdir(path))

	# load state dict
	fname_pt = [
		f for f in files if
		f.split('.')[-1] == 'pt'
	]
	if checkpoint == -1:
		fname_pt = fname_pt[-1]
	else:
		fname_pt = next(
			f for f in fname_pt if
			checkpoint == _chkpt(f)
		)
	state_dict = pjoin(path, fname_pt)
	state_dict = torch.load(state_dict, 'cpu')
	ema = state_dict['model_ema'] is not None
	model.load_state_dict(
		state_dict=state_dict['model'],
		strict=strict,
	)

	# set chkpt_dir & timestamp
	model.chkpt_dir = path
	timestamp = state_dict['metadata'].get('timestamp')
	if timestamp is not None:
		model.timestamp = timestamp

	# load trainer
	fname = next(
		f for f in files if
		f.split('.')[-1] == 'json'
	)
	with open(pjoin(path, fname), 'r') as f:
		cfg_train = json.load(f)
	fname = fname.split('.')[0]
	if fname == 'ConfigTrainVAE':
		from main.train_vae import TrainerVAE
		cfg_train = ConfigTrainVAE(**cfg_train)
		trainer = TrainerVAE(
			model=model,
			cfg=cfg_train,
			device=device,
			verbose=verbose,
			**kwargs,
		)
	else:
		raise NotImplementedError(fname)

	if ema:
		trainer.model_ema.load_state_dict(
			state_dict=state_dict['model_ema'],
			strict=strict,
		)
		if timestamp is not None:
			trainer.model_ema.timestamp = timestamp

	if strict:
		trainer.optim.load_state_dict(
			state_dict['optim'])
		trainer.scaler.load_state_dict(
			state_dict['scaler'])
		if trainer.optim_schedule is not None:
			trainer.optim_schedule.load_state_dict(
				state_dict.get('scheduler', {}))
	stats = state_dict['metadata'].pop('stats', {})
	trainer.stats.update(stats)
	metadata = {
		**state_dict['metadata'],
		'file': fname_pt,
	}
	return trainer, metadata


class Initializer:
	def __init__(self, dist: str, **kwargs):
		self.mode = 'pytorch'
		try:
			dist = getattr(dists, dist)
			kwargs = filter_kwargs(dist, kwargs)
		except AttributeError:
			dist = getattr(sp_stats, dist)
			self.mode = 'scipy'
		self.dist = dist(**kwargs)

	@torch.inference_mode()
	def apply(self, weight: torch.Tensor):
		if self.mode == 'scipy':
			values = self.dist.rvs(tuple(weight.shape))
			values = torch.tensor(values, dtype=torch.float)
		else:
			values = self.dist.sample(weight.shape)
		weight.data.copy_(values.to(weight.device))
		return


class Module(nn.Module):
	def __init__(self, cfg, verbose: bool = False):
		super(Module, self).__init__()
		self.cfg = cfg
		self.chkpt_dir = None
		self.timestamp = now(True)
		self.verbose = verbose

	def create_chkpt_dir(self, fit_name: str = None):
		chkpt_dir = '_'.join([
			fit_name if fit_name else
			f"seed-{self.cfg.seed}",
			f"({self.timestamp})",
		])
		chkpt_dir = pjoin(
			self.cfg.mods_dir,
			chkpt_dir,
		)
		os.makedirs(chkpt_dir, exist_ok=True)
		self.chkpt_dir = chkpt_dir
		return

	def save(
			self,
			checkpoint: int = -1,
			name: str = None,
			path: str = None, ):
		path = path if path else self.chkpt_dir
		name = name if name else type(self).__name__
		fname = '-'.join([
			name,
			f"{checkpoint:04d}",
			f"({now(True)}).pt",
		])
		fname = pjoin(path, fname)
		torch.save(self.state_dict(), fname)
		return fname


def _chkpt(f):
	return int(f.split('_')[0].split('-')[-1])


def _sort_fn(f: str):
	f = f.split('(')[-1].split(')')[0]
	ymd, hm = f.split(',')
	yy, mm, dd = ymd.split('_')
	h, m = hm.split(':')
	yy, mm, dd, h, m = map(
		lambda s: int(s),
		[yy, mm, dd, h, m],
	)
	x = (
		yy * 1e8 +
		mm * 1e6 +
		dd * 1e4 +
		h * 1e2 +
		m
	)
	return x
