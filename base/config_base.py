from utils.generic import *
_SCHEDULER_CHOICES = ['cosine', 'exponential', 'step', 'cyclic', None]
_OPTIM_CHOICES = ['adamax_fast', 'adamax', 'adam', 'adamw', 'radam', 'sgd']


class BaseConfig(object):
	def __init__(
			self,
			seed: int = 0,
			save: bool = True,
			data_dir: str = 'Datasets',
			base_dir: str = 'Projects/PoissonVAE',
	):
		super(BaseConfig, self).__init__()
		self._set_seeds(seed)
		name = self.name()
		# setup directories
		self.base_dir, self.data_dir = map(
			add_home, [base_dir, data_dir])
		self.runs_dir = pjoin(self.base_dir, 'runs', name)
		self.mods_dir = pjoin(self.base_dir, 'models', name)
		self.results_dir = pjoin(self.base_dir, 'results')
		if save:
			self._mkdirs()
			self.save()

	def name(self):
		name = self._name()
		if self.seed > 0:
			name = f"{name}_seed-{self.seed}"
		return name

	def _name(self):
		raise NotImplementedError

	def save(self):
		raise NotImplementedError

	def _save(self, **kwargs):
		_save_config(self, str(self.mods_dir), **kwargs)

	def get_all_dirs(self):
		dirs = {k: getattr(self, k) for k in dir(self) if '_dir' in k}
		dirs = filter(lambda x: isinstance(x[1], str), dirs.items())
		return dict(dirs)

	def _mkdirs(self):
		for _dir in self.get_all_dirs().values():
			os.makedirs(_dir, exist_ok=True)

	def _set_seeds(self, seed):
		min_val = np.iinfo(np.uint32).min
		max_val = np.iinfo(np.uint32).max
		assert min_val <= seed <= max_val

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		os.environ["SEED"] = str(seed)
		np.random.seed(seed)
		random.seed(seed)
		self.seed = seed
		return


class BaseConfigTrain(object):
	@validate_choices(_SCHEDULER_CHOICES, 'scheduler_type')
	@validate_choices(_OPTIM_CHOICES, 'optimizer')
	def __init__(
			self,
			lr: float,
			epochs: int,
			batch_size: int,
			warm_restart: int,
			warmup_portion: float,
			optimizer: str,
			scheduler_type: str,
			ema_rate: float = None,
			grad_clip: float = 1000,
			use_amp: bool = False,
			chkpt_freq: int = 50,
			eval_freq: int = 10,
			log_freq: int = 20,
	):
		super(BaseConfigTrain, self).__init__()
		assert warm_restart >= 0
		assert warmup_portion >= 0.0

		self.lr = lr
		self.epochs = epochs
		self.batch_size = batch_size

		self.warm_restart = warm_restart
		self.warmup_portion = warmup_portion

		self.optimizer = optimizer
		self.scheduler_type = scheduler_type
		self.optimizer_kws = None
		self.scheduler_kws = None
		self.set_optim_kws()
		self.set_scheduler_kws()

		self.ema_rate = ema_rate
		self.grad_clip = grad_clip
		self.chkpt_freq = chkpt_freq
		self.eval_freq = eval_freq
		self.log_freq = log_freq
		self.use_amp = use_amp

	def name(self):
		raise NotImplementedError

	def save(self, save_dir: str):
		raise NotImplementedError

	def _save(self, save_dir: str, **kwargs):
		_save_config(self, save_dir, **kwargs)

	def set_optim_kws(self, **kwargs):
		if self.optimizer == 'sgd':
			defaults = {
				'momentum': 0.9,
				'weight_decay': 0.0,
				'nesterov': True,
			}
		else:
			defaults = {
				'betas': (0.9, 0.999),
				'weight_decay': 0.0,
				'eps': 1e-8,
			}
		self.optimizer_kws = setup_kwargs(
			defaults=defaults, kwargs=kwargs)
		return

	def set_scheduler_kws(self, **kwargs):
		eta_min = 1e-5
		period = self.epochs * (1 - self.warmup_portion)
		period /= (2 * self.warm_restart + 1)
		if self.scheduler_type == 'cosine':
			defaults = {
				'T_max': period,
				'eta_min': eta_min,
			}
		elif self.scheduler_type == 'exponential':
			defaults = {
				'gamma': 0.9,
				'eta_min': eta_min,
			}
		elif self.scheduler_type == 'step':
			defaults = {
				'gamma': 0.1,
				'step_size': 10,
			}
		elif self.scheduler_type == 'cyclic':
			defaults = {
				'max_lr': self.lr,
				'base_lr': eta_min,
				'mode': 'exp_range',
				'step_size_up': period,
				'step_size': 10,
				'gamma': 0.9,
			}
		elif self.scheduler_type is None:
			defaults = {}
		else:
			raise NotImplementedError(self.scheduler_type)
		self.scheduler_kws = setup_kwargs(
			defaults=defaults, kwargs=kwargs)
		return


def _save_config(
		obj,
		save_dir: str,
		with_base: bool = True,
		overwrite: bool = True,
		verbose: bool = False, ):
	fname = type(obj).__name__
	file = pjoin(save_dir, f"{fname}.json")
	if os.path.isfile(file) and not overwrite:
		return
	save_obj(
		obj=obj_attrs(obj, with_base),
		file_name=fname,
		save_dir=save_dir,
		verbose=verbose,
		mode='json',
	)
	return
