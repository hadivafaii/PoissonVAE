from base.utils_model import *
import subprocess
import wandb


def _get_git_root() -> Optional[str]:
	"""
	Finds git root relative to THIS file, not the current working directory.
	This ensures it works even if run from a notebook outside the repo.
	"""
	try:
		# Anchor the search to the location of train_base.py
		anchor_dir = os.path.dirname(os.path.abspath(__file__))

		git_root = subprocess.check_output(
			['git', 'rev-parse', '--show-toplevel'],
			stderr=subprocess.DEVNULL,
			cwd=anchor_dir,  # <--- This is the key fix
		).decode('utf-8').strip()
		return git_root
	except (subprocess.CalledProcessError, FileNotFoundError, OSError):
		return None



class BaseTrainer(object):
	def __init__(
			self,
			model: Module,
			cfg: Any,
			device: str = 'cpu',
			shuffle: bool = True,
			verbose: bool = False,
	):
		super(BaseTrainer, self).__init__()
		self.cfg = cfg
		self.shuffle = shuffle
		self.verbose = verbose
		self.device = torch.device(device)
		self.model = model.to(self.device).eval()
		self.stats = collections.defaultdict(dict)
		self.scaler = torch.amp.GradScaler(
			enabled=self.cfg.use_amp)
		self.model_ema = None
		self.ema_rate = None
		self.n_iters = None
		self.pbar = None

		self.wandb_run = None
		self.logger = None
		self.dl_trn = None
		self.dl_vld = None
		self.dl_tst = None
		self.setup_data()

		self.optim = None
		self.optim_schedule = None
		self.setup_optim()

		if self.verbose:
			tot = sum([
				p.nelement() for p in
				self.model.parameters()
			])
			if tot // 1e6 > 0:
				tot = f"{tot / 1e6:0.1f} M"
			elif tot // 1e3 > 0:
				tot = f"{tot / 1e3:0.1f} K"
			print(f"\n# params: {tot}")

	def train(
			self,
			fit_name: str = None,
			epochs: Union[int, range] = None,
			fresh_fit: bool = True,
			save: bool = True, ):
		epochs = epochs if epochs else self.cfg.epochs
		assert isinstance(epochs, (int, range)), "allowed: {int, range}"
		epochs = range(epochs) if isinstance(epochs, int) else epochs
		fit_name = fit_name if fit_name else self.cfg.name()

		# Init wandb (no effect when no_wandb=True)
		self._init_wandb_run(fit_name, fresh_fit)

		if save:
			if fresh_fit or self.model.chkpt_dir is None:
				self.model.create_chkpt_dir(fit_name)
			if fresh_fit:
				self.cfg.save(self.model.chkpt_dir)

		if fresh_fit and self.model.chkpt_dir is not None:
			self.stats.clear()
			self.logger = make_logger(
				name=type(self).__name__,
				path=self.model.chkpt_dir,
				level=logging.WARNING,
			)

		if self.cfg.scheduler_type == 'cosine' and fresh_fit:
			self.optim_schedule.T_max *= len(self.dl_trn)

		self.pbar = tqdm(
			iterable=epochs,
			dynamic_ncols=True,
			leave=True,
			position=0,
		)
		for epoch in self.pbar:
			avg_loss = self.iteration(epoch)
			msg = ', '.join([
				f"epoch # {epoch + 1:d}",
				f"avg loss: {avg_loss:3f}",
			])
			self.pbar.set_description(msg)
			if not save:
				continue
			if (epoch + 1) % self.cfg.chkpt_freq == 0:
				self.save(
					checkpoint=epoch + 1,
					path=self.model.chkpt_dir,
				)
			if (epoch + 1) % self.cfg.eval_freq == 0:
				gstep = (epoch + 1) * len(self.dl_trn)
				_ = self.validate(gstep)

		if self.wandb_run is not None:
			wandb.finish()
		return

	def iteration(self, epoch: int = 0):
		raise NotImplementedError

	@torch.inference_mode()
	def validate(self, epoch: int = None):
		raise NotImplementedError

	def setup_data(self):
		raise NotImplementedError

	def _init_wandb_run(self, fit_name: str, fresh_fit: bool):
		no_wandb = getattr(self.cfg, 'no_wandb', False)
		if no_wandb:
			return

		# 1. Prepare Config
		run_config = {}
		if hasattr(self.model, 'cfg'):
			run_config.update(vars(self.model.cfg))
		if hasattr(self, 'cfg'):
			run_config.update(vars(self.cfg))

		# 2. Init WandB
		self.wandb_run = wandb.init(
			project=getattr(self.cfg, 'wandb_project', 'PoissonVAE'),
			entity=getattr(self.cfg, 'wandb_entity', None),
			name=fit_name,
			config=run_config,
			dir=self.model.cfg.runs_dir if
			hasattr(self.model.cfg, 'runs_dir')
			else None,
			resume="allow",
			tags=["resume"] if not fresh_fit else [],
		)

		# 3. Find git root for code saving
		git_root = _get_git_root()

		# 4. Log Code Snapshot (only on fresh start)
		if fresh_fit and git_root is not None:
			if self.verbose:
				print(f"Saving code snapshot from: {git_root}")
			# noinspection PyArgumentList
			self.wandb_run.log_code(
				root=git_root,
				include_fn=lambda path: (
					path.endswith(".py") or
					path.endswith(".yaml") or
					path.endswith(".yml") or
					path.endswith(".json") or
					path.endswith(".md") or
					path.endswith(".txt")),
				exclude_fn=lambda path: (
					"wandb" in path or
					"__pycache__" in path or
					".git" in path or
					"node_modules" in path or
					".egg-info" in path),
			)
		return

	def swap_model(self, new_model, full: bool = False):
		self.model = new_model.to(self.device).eval()
		if full:
			self.setup_data()
			self.setup_optim()
		return

	def select_model(self, ema: bool = False):
		if ema:
			assert self.model_ema is not None
			return self.model_ema.eval()
		return self.model.eval()

	def reset_model(self):
		raise NotImplementedError

	def update_ema(self):
		if self.model_ema is None:
			return
		looper = zip(
			self.model.parameters(),
			self.model_ema.parameters(),
		)
		for p1, p2 in looper:
			p2.data.mul_(self.ema_rate)
			p2.data.add_(p1.data.mul(1 - self.ema_rate))
		return

	def parameters(self, requires_grad: bool = True):
		if requires_grad:
			return filter(
				lambda p: p.requires_grad,
				self.model.parameters(),
			)
		else:
			return self.model.parameters()

	def save(self, path: str, checkpoint: int = None):
		if checkpoint is not None:
			global_step = checkpoint * len(self.dl_trn)
		else:
			global_step = None
		state_dict = {
			'metadata': {
				'timestamp': self.model.timestamp,
				'checkpoint': checkpoint,
				'global_step': global_step,
				'stats': self.stats,
				'root': path},
			'model': self.model.state_dict(),
			'model_ema': self.model_ema.state_dict()
			if self.model_ema is not None else None,
			'optim': self.optim.state_dict(),
			'scaler': self.scaler.state_dict(),
			'scheduler': self.optim_schedule.state_dict()
			if self.optim_schedule is not None else None,
		}
		fname = '+'.join([
			type(self.model).__name__,
			type(self).__name__],
		)
		if checkpoint is not None:
			fname = '-'.join([
				fname,
				f"{checkpoint:04d}"
			])
		fname = f"{fname}_({now(True)}).pt"
		fname = pjoin(path, fname)
		torch.save(state_dict, fname)
		return

	def setup_optim(self):
		# optimzer
		kws = dict(
			params=self.parameters(),
			lr=self.cfg.lr,
			**self.cfg.optimizer_kws,
		)
		if self.cfg.optimizer == 'adamax':
			self.optim = torch.optim.Adamax(**kws)
		elif self.cfg.optimizer == 'adamax_fast':
			from .adamax import Adamax
			self.optim = Adamax(**kws)
		elif self.cfg.optimizer == 'adam':
			self.optim = torch.optim.Adam(**kws)
		elif self.cfg.optimizer == 'adamw':
			self.optim = torch.optim.AdamW(**kws)
		elif self.cfg.optimizer == 'radam':
			self.optim = torch.optim.RAdam(**kws)
		else:
			raise NotImplementedError(self.cfg.optimizer)

		# scheduler
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'exponential':
			self.optim_schedule = torch.optim.lr_scheduler.ExponentialLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'step':
			self.optim_schedule = torch.optim.lr_scheduler.StepLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'cyclic':
			self.optim = torch.optim.SGD(
				params=self.parameters(),
				lr=self.cfg.lr,
				momentum=0.9,
				weight_decay=self.cfg.optimizer_kws.get('weight_decay', 0),
			)
			self.optim_schedule = torch.optim.lr_scheduler.CyclicLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type is None:
			self.optim_schedule = None
		else:
			raise NotImplementedError(self.cfg.scheduler_type)
		return

	def info(self, filter_dirs: bool = True):
		txt = '\n'.join([
			'_' * 100,
			'[INFO] Checkpoint Directory:',
			f"{self.model.chkpt_dir}\n\n",
		])
		# model
		cfg_model = vars(self.model.cfg)
		if filter_dirs:
			cfg_model = {
				key: v for key, v
				in cfg_model.items()
				if '_dir' not in key
			}
		txt += f"{'_' * 100}\n[INFO] model name: {self.model.cfg.name()}"
		txt += f"\n[INFO] model cfg:\n{json.dumps(cfg_model, indent=4)}"
		# trainer
		txt += f"\n\n{'_' * 100}\n[INFO] trainer name: {self.cfg.name()}"
		txt += f"\n[INFO] trainer cfg:\n{json.dumps(vars(self.cfg), indent=4)}"
		return txt

	def to(self, x, dtype=torch.float32) -> Union[torch.Tensor, List[torch.Tensor]]:
		kws = dict(device=self.device, dtype=dtype)
		if isinstance(x, (tuple, list)):
			return [
				e.to(**kws) if torch.is_tensor(e)
				else torch.tensor(e, **kws)
				for e in x
			]
		else:
			if torch.is_tensor(x):
				return x.to(**kws)
			else:
				return torch.tensor(x, **kws)
