from base.train_base import *
from base.dataset import make_dataset
from .vae import (
	MODEL_CLASSES, BaseVAE,
	PoissonVAE, GaussianVAE,
	CategoricalVAE, LaplaceVAE,
)
from figures.imgs import make_grid
from analysis.eval import (
	sparse_score,
	knn_analysis,
	model2temp,
	model2key,
)


class _BaseTrainerVAE(BaseTrainer):
	def __init__(
			self,
			model: BaseVAE,
			cfg: ConfigTrainVAE,
			**kwargs,
	):
		super(_BaseTrainerVAE, self).__init__(
			model=model, cfg=cfg, **kwargs)
		self.n_iters = self.cfg.epochs * len(self.dl_trn)
		if self.cfg.kl_anneal_cycles == 0:
			self.betas = beta_anneal_linear(
				n_iters=self.n_iters,
				beta=self.cfg.kl_beta,
				anneal_portion=self.cfg.kl_anneal_portion,
				constant_portion=self.cfg.kl_const_portion,
				min_beta=self.cfg.kl_beta_min,
			)
		else:
			betas = beta_anneal_cosine(
				n_iters=self.n_iters,
				n_cycles=self.cfg.kl_anneal_cycles,
				portion=self.cfg.kl_anneal_portion,
				start=np.arccos(
					1 - 2 * self.cfg.kl_beta_min
					/ self.cfg.kl_beta) / np.pi,
				beta=self.cfg.kl_beta,
			)
			beta_cte = int(np.round(self.cfg.kl_const_portion * self.n_iters))
			beta_cte = np.ones(beta_cte) * self.cfg.kl_beta_min
			self.betas = np.insert(betas, 0, beta_cte)[:self.n_iters]
		if self.cfg.lambda_anneal:
			self.wd_coeffs = beta_anneal_linear(
				n_iters=self.n_iters,
				beta=self.cfg.lambda_norm,
				anneal_portion=self.cfg.kl_anneal_portion,
				constant_portion=self.cfg.kl_const_portion,
				min_beta=self.cfg.lambda_init,
			)
		else:
			self.wd_coeffs = np.ones(self.n_iters)
			self.wd_coeffs *= self.cfg.lambda_norm
		if self.cfg.temp_anneal_portion > 0.0:
			kws = dict(
				n_iters=self.n_iters,
				portion=self.cfg.temp_anneal_portion,
				t0=self.cfg.temp_start,
				t1=self.cfg.temp_stop,
			)
			if self.cfg.temp_anneal_type == 'lin':
				self.temperatures = temp_anneal_linear(**kws)
			elif self.cfg.temp_anneal_type == 'exp':
				self.temperatures = temp_anneal_exp(**kws)
			else:
				raise ValueError(self.cfg.temp_anneal_type)
		else:
			self.temperatures = np.ones(self.n_iters)
			self.temperatures *= self.cfg.temp_stop

	def iteration(self, epoch: int = 0):
		raise NotImplementedError

	@torch.inference_mode()
	def validate(self, epoch: int = None):
		raise NotImplementedError

	@torch.inference_mode()
	def sample(self, n_samples: int, temp: float):
		raise NotImplementedError

	def find_dead_neurons(self, kl: np.ndarray = None):
		if kl is None:
			loss = self.validate()[1]
			kl = loss['kl_diag']
		eps = np.finfo(kl.dtype).eps
		kl = np.maximum(kl, eps)
		# custom rules
		rules = {
			('pm', 1e-2): (
				self.model.cfg.type == 'poisson' and
				self.model.cfg.dataset == 'MNIST'),
			('ll', 1e-1): (
				self.model.cfg.type == 'laplace' and
				self.model.cfg.enc_type == 'lin'),
			('gl', 1e-1): (
				self.model.cfg.type == 'gaussian' and
				self.model.cfg.enc_type == 'lin' and
				self.model.cfg.dataset != 'CIFAR10-PATCHES'),
			('glc', 85e-3): (
				self.model.cfg.type == 'gaussian' and
				self.model.cfg.enc_type == 'lin' and
				self.model.cfg.dataset == 'CIFAR10-PATCHES'),
		}
		for (_, thres), r in rules.items():
			if r:
				return kl < thres

		if self.model.cfg.type == 'categorical':
			dead = self.model.find_dead_neurons(2)
		else:
			if self.model.cfg.enc_type == 'lin':
				dead = self.model.find_dead_neurons(8)
				order = np.argsort(kl)
				idx = np.argmax(~dead[order])
				dead = np.zeros(len(dead))
				dead[order[:idx]] = 1
			else:
				log_kl = np.log(kl)
				bins = np.linspace(
					start=np.nanmin(log_kl),
					stop=np.nanmax(log_kl),
					num=len(log_kl) * 10,
				)
				hist, _ = np.histogram(
					log_kl, bins=bins)

				idx = find_last_contiguous_zeros(
					hist > 0, len(log_kl) * 2)
				dead = log_kl < bins[idx]
			dead = dead.astype(bool) | (kl < 3e-4)
		return dead.astype(bool)

	def setup_data(self, gpu: bool = True):
		# create datasets
		device = self.device if gpu else None
		trn, vld, tst = make_dataset(
			dataset=self.model.cfg.dataset,
			load_dir=self.model.cfg.data_dir,
			device=device,
		)
		# create dataloaders
		kws = dict(
			batch_size=self.cfg.batch_size,
			drop_last=self.shuffle,
			shuffle=self.shuffle,
		)
		self.dl_trn = torch.utils.data.DataLoader(trn, **kws)
		kws.update({'drop_last': False, 'shuffle': False})
		self.dl_vld = torch.utils.data.DataLoader(vld, **kws)
		self.dl_tst = torch.utils.data.DataLoader(tst, **kws)
		return

	@torch.inference_mode()
	def show_recon(
			self,
			t: float = None,
			n_samples: int = 16,
			display: bool = True,
			**kwargs, ):
		defaults = dict(
			dpi=100,
			figsize=(8, 1),
			cmap='Greys_r',
			normalize=False,
			pad=0,
		)
		kwargs = setup_kwargs(defaults, kwargs)

		x = next(iter(self.dl_vld))[0]
		x = self.to(x)[:n_samples]

		if t is not None:
			t_original = self.model.temp.item()
			self.model.update_t(t)
		else:
			t_original = None

		# get recon
		y = self.model(x)[-1]

		if t is not None:
			self.model.update_t(t_original)

		g2p = make_grid(
			torch.cat([x, y]),
			grid_size=(2, n_samples),
			pad=kwargs['pad'],
			normalize=kwargs['normalize'],
		)

		fig, ax = create_figure(
			figsize=kwargs['figsize'],
			dpi=kwargs['dpi'],
		)
		ax.imshow(g2p, cmap=kwargs['cmap'])
		remove_ticks(ax)
		if display:
			plt.show()
		else:
			plt.close()
		return fig, ax

	@torch.inference_mode()
	def show_samples(
			self,
			t: float = None,
			nrows: int = 10,
			display: bool = True,
			**kwargs, ):
		defaults = dict(
			dpi=100,
			figsize=(3.5, 3.5),
			cmap='Greys_r',
			normalize=False,
			pad=0,
		)
		kwargs = setup_kwargs(defaults, kwargs)

		# sample
		x, _ = self.sample(nrows ** 2, t)
		g2p = make_grid(
			x=x[:nrows ** 2],
			grid_size=nrows,
			pad=kwargs['pad'],
			normalize=kwargs['normalize'],
		)
		fig, ax = create_figure(
			figsize=kwargs['figsize'],
			dpi=kwargs['dpi'],
		)
		ax.imshow(g2p, cmap=kwargs['cmap'])
		remove_ticks(ax)
		if display:
			plt.show()
		else:
			plt.close()
		return fig, ax

	def show_schedules(self):
		fig, axes = create_figure(
			nrows=1, ncols=3,
			figsize=(6.5, 1.6),
			layout='constrained',
		)
		# temp
		axes[0].plot(self.temperatures, color='k', lw=3)
		axes[0].axhline(
			self.cfg.temp_start,
			color='g',
			ls='--',
			label=f"t0 = {self.cfg.temp_start:0.2g}",
		)
		axes[0].axhline(
			self.cfg.temp_stop,
			color='r',
			ls='--',
			label=f"t1 = {self.cfg.temp_stop:0.2g}",
		)
		# beta
		axes[1].plot(self.betas, color='C0', lw=3)
		axes[1].axhline(
			self.cfg.kl_beta,
			color='g',
			ls='--',
			label=r"$\beta = $" + f"{self.cfg.kl_beta:0.2g}",
		)
		# lamb
		axes[2].plot(self.wd_coeffs, color='C4', lw=3)
		axes[2].axhline(
			self.cfg.lambda_norm,
			color='g',
			ls='--',
			label=r"$\lambda = $" + f"{self.cfg.lambda_norm:0.2g}",
		)

		add_legend(axes, fontsize=10)
		for ax in axes.flat:
			ax.ticklabel_format(
				axis='x',
				style='sci',
				scilimits=(0, 0),
			)
		plt.show()
		return

	def reset_model(self):
		raise NotImplementedError


class TrainerVAE(_BaseTrainerVAE):
	def __init__(
			self,
			model: Union[
				PoissonVAE,
				GaussianVAE,
				CategoricalVAE,
				LaplaceVAE],
			cfg: ConfigTrainVAE,
			**kwargs,
	):
		super(TrainerVAE, self).__init__(
			model=model, cfg=cfg, **kwargs)
		self._init_fun()

		if self.cfg.ema_rate is not None:
			model_class = MODEL_CLASSES[self.model.cfg.type]
			model_ema = model_class(self.model.cfg)
			self.model_ema = model_ema.to(self.device).eval()
			self.ema_rate = self.to(self.cfg.ema_rate)

	def iteration(self, epoch: int = 0):
		self.model.train()

		nelbo = AverageMeter()
		grads = AverageMeter()
		r_max = AverageMeter()
		perdim_kl = AverageMeter()
		perdim_mse = AverageMeter()

		for i, (x, *_) in enumerate(self.dl_trn):
			gstep = epoch * len(self.dl_trn) + i
			# warm-up lr?
			progress = gstep / self.n_iters
			is_warming_up = progress < self.cfg.warmup_portion
			if is_warming_up:
				lr = (
					self.cfg.lr * progress /
					self.cfg.warmup_portion
				)
				for param_group in self.optim.param_groups:
					param_group['lr'] = lr
			# send to device
			if x.device != self.device:
				x = self.to(x)
			# zero grad
			self.optim.zero_grad(set_to_none=True)
			# forward + loss
			with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
				self.model.update_t(self.temperatures[gstep])
				recon_batch, kl, dist = self._fun(x)
				kl_batch = torch.sum(kl, dim=1)
				kl_diag = torch.mean(kl, dim=0)

				loss = recon_batch + self.betas[gstep] * kl_batch
				loss = torch.mean(loss)

				# add regularization
				if self.wd_coeffs[gstep] > 0:
					loss_w = self.model.loss_weight()
					if loss_w is not None:
						loss += self.wd_coeffs[gstep] * loss_w
				else:
					loss_w = None

			# backward
			self.scaler.scale(loss).backward()
			self.scaler.unscale_(self.optim)
			# clip grad
			if self.cfg.grad_clip is not None:
				if is_warming_up:
					max_norm = self.cfg.grad_clip * 3
				else:
					max_norm = self.cfg.grad_clip
				grad_norm = nn.utils.clip_grad_norm_(
					parameters=self.parameters(),
					max_norm=max_norm,
				).item()
				grads.update(grad_norm)
				self.stats['grad'][gstep] = grad_norm
				if grad_norm > self.cfg.grad_clip:
					self.stats['loss'][gstep] = loss.item()
			# update average meters & stats
			with torch.inference_mode():
				_v = recon_batch + kl_batch
				_v = _v.mean().item()
				nelbo.update(_v)
				_v = recon_batch.mean().item()
				_v /= self.model.cfg.input_sz ** 2
				perdim_mse.update(_v)
				perdim_kl.update(kl_diag.mean().item())
				if self.model.cfg.type == 'poisson':
					r_max.update(dist.rate.max().item())

			# step
			self.scaler.step(self.optim)
			self.scaler.update()
			self.update_ema()
			# optim schedule
			cond_schedule = (
				not is_warming_up and
				self.optim_schedule is not None
			)
			if cond_schedule:
				self.optim_schedule.step()

			# save more stats
			current_lr = self.optim.param_groups[0]['lr']
			self.stats['lr'][gstep] = current_lr
			self.stats['temp'][gstep] = dist.temp.item()
			if self.model.cfg.type == 'poisson':
				self.stats['r_max'][gstep] = r_max.avg
				self.stats['n_exp'][gstep] = self.model.n_exp.item()

			# WANDB LOGGING
			cond_write = (
				gstep > 0 and
				self.wandb_run is not None and
				gstep % self.cfg.log_freq == 0
			)
			if not cond_write:
				continue

			to_write = {
				'coeffs/beta': self.betas[gstep],
				'coeffs/temp': self.temperatures[gstep],
				'coeffs/lr': self.optim.param_groups[0]['lr'],
			}
			if self.wd_coeffs[gstep] > 0:
				to_write['coeffs/reg_coeff'] = self.wd_coeffs[gstep]
			if self.model.cfg.type == 'poisson':
				to_write.update({
					'coeffs/r_max': r_max.avg,
					'coeffs/n_exp': self.model.n_exp,
				})
			to_write.update({
				'train/loss_kl': torch.mean(kl_batch).item(),
				'train/loss_mse': torch.mean(recon_batch).item(),
				'train/nelbo_avg': nelbo.avg,
				'train/perdim_kl': perdim_kl.avg,
				'train/perdim_mse': perdim_mse.avg,
				'train/weight_reg': 0 if loss_w is None
				else loss_w.item(),
			})
			if self.cfg.grad_clip is not None:
				to_write['train/grad_norm'] = grads.avg
			n_active = torch.sum(kl_diag > 0.01).item()
			to_write['train/n_active'] = n_active
			ratio = n_active / self.model.cfg.n_latents
			to_write['train/n_active_ratio'] = ratio

			# Log to WandB
			wandb.log(to_write, step=gstep)

			# reset avg meters
			if gstep % 100 == 0:
				grads.reset()
				nelbo.reset()

		# end of epock n_exp update
		if self.model.cfg.type == 'poisson':
			self.model.update_n(r_max.avg)

		return nelbo.avg

	@torch.inference_mode()
	def validate(
			self,
			gstep: int = None,
			n_samples: int = 4096,
			**kwargs, ):
		# forward
		temp = model2temp(self.model.cfg.type)
		kwargs = setup_kwargs({'temp': temp}, kwargs)
		data, loss, etc = self.forward('vld', **kwargs)

		if gstep is not None:
			# prep to write
			to_write = {
				f"eval/{k}": v.mean()
				for k, v in loss.items()
			}

			# overall sparse coding performance
			r2_score = max(0.0, float(np.mean(loss['r2'])))
			portion_zeros = float(np.mean(data['z'] == 0))
			overall_performance = np.sqrt(
				(1 - r2_score) ** 2 +
				(1 - portion_zeros) ** 2
			) / np.sqrt(2.0)

			# other sparsity metrics
			lifetime, population, _ = sparse_score(
				z=data['z'], cutoff=None)
			to_write.update({
				'sprs/overall': overall_performance,
				'sprs/portion_zeros': portion_zeros,
				'sprs/lifetime': np.nanmean(lifetime),
				'sprs/population': np.nanmean(population),
			})

			# knn analysis
			if self.model.cfg.dataset == 'MNIST':
				key = model2key(self.model.cfg.type)
				_, df_summary = knn_analysis(
					x=flatten_np(etc.get(key), start_dim=1),
					y=tonp(self.dl_vld.dataset.tensors[1]).astype(int),
					seed=self.model.cfg.seed,
					sizes=[200, 1000],
					n_iter=500 if (  # last epoch
						gstep ==
						self.cfg.epochs *
						len(self.dl_trn)
					) else 20,
				)
				for size, accuracy in df_summary['mean'].items():
					to_write[f'knn/size={size}'] = accuracy

			# Log scalars to WandB (only if wandb is active)
			if self.wandb_run is not None:
				wandb.log(to_write, step=gstep)

			for k, v in to_write.items():
				self.stats[k][gstep] = v

			# Log figures to WandB
			if self.model.cfg.type == 'categorical':
				order = np.argsort(etc['logits'].mean(0).ravel())
			else:
				order = np.argsort(loss['kl_diag'].ravel())
			self.log_figs_to_wandb(gstep, kwargs.get('temp'), order)

		return data, loss, etc

	def forward(
			self,
			dl_name: str,
			temp: float = None,
			use_ema: bool = False,
			full_data: bool = False, ):
		assert dl_name in ['trn', 'vld', 'tst']
		dl = getattr(self, f"dl_{dl_name}")
		if dl is None:
			return None
		model = self.select_model(use_ema)
		if temp is None:
			temp = model.temp

		r2, mse, kl, kl_diag = [], [], [], []
		x_all, y_all, z_all, g_all = [], [], [], []
		etc = collections.defaultdict(list)

		for x in iter(dl):
			if len(x) == 2:
				x, g = x
			else:
				g = None
			if isinstance(x, (tuple, list)):
				x = x[0]
			if x.device != self.device:
				x = self.to(x)
			if self.model.cfg.type == 'poisson':
				dist, log_dr, z, y = model.xtract_ftr(x, temp)
				_kl = model.loss_kl(log_dr)
				etc['log_dr'].append(tonp(log_dr))
				etc['r*dr'].append(tonp(dist.rate))
			elif self.model.cfg.type in [
					'gaussian', 'laplace', 'categorical']:
				dist, z, y = model.xtract_ftr(x, temp)
				_kl = model.loss_kl(dist)
				if self.model.cfg.type == 'categorical':
					etc['logits'].append(tonp(dist.logits))
				else:
					etc['loc'].append(tonp(dist.loc))
					etc['scale'].append(tonp(dist.scale))
			else:
				raise ValueError(self.model.cfg.type)
			# data
			if full_data or dl_name == 'trn':
				x_all.append(tonp(x))
				y_all.append(tonp(y))
				if g is not None:
					g_all.append(tonp(g))
			z_all.append(tonp(z))
			# loss
			r2.append(tonp(compute_r2(
				true=x.flatten(start_dim=1),
				pred=y.flatten(start_dim=1),
			)))
			mse.append(tonp(model.loss_recon(y, x)))
			kl.append(tonp(torch.sum(_kl, dim=1)))
			kl_diag.append(tonp(torch.mean(
				_kl, dim=0, keepdim=True)))

		x, y, z, g, r2, mse, kl, kl_diag = cat_map(
			[x_all, y_all, z_all, g_all, r2, mse, kl, kl_diag])
		data = {'x': x, 'y': y, 'z': z, 'g': g}
		loss = {
			'r2': r2,
			'mse': mse,
			'kl': kl,
			'elbo': mse + kl,
			'kl_diag': kl_diag.mean(0),
		}
		etc = {k: np.concatenate(v) for k, v in etc.items()}
		return data, loss, etc

	@torch.inference_mode()
	def mse_map(self):
		mse_map = []
		for x, *_ in iter(self.dl_vld):
			dist = self.model.infer(x)
			if isinstance(dist, tuple):
				dist = dist[0]
			y = self.model.decode(dist.mean)
			mse_map.append(self.model.loss_recon(y, x))
		mse_map = tonp(torch.cat(mse_map))
		return mse_map

	@torch.inference_mode()
	def sample(
			self,
			n_samples: int = 4096,
			temp: float = None,
			use_ema: bool = False, ):
		model = self.select_model(use_ema)
		if temp is None:
			temp = model.temp
		num = n_samples / self.cfg.batch_size
		num = int(np.ceil(num))
		x_sample, z_sample = [], []
		tot = 0
		for _ in range(num):
			n = self.cfg.batch_size
			if tot + self.cfg.batch_size > n_samples:
				n = n_samples - tot
			_x, _z = model.sample(n, temp)
			x_sample.append(tonp(_x))
			z_sample.append(tonp(_z))
			tot += self.cfg.batch_size
		x_sample, z_sample = cat_map([
			x_sample, z_sample])
		return x_sample, z_sample

	@torch.inference_mode()
	def log_figs_to_wandb(
			self,
			gstep: int,
			temp: float,
			order: Sequence[int] = None, ):
		if self.wandb_run is None:
			return None

		freq = max(10, self.cfg.eval_freq * 5)
		ep = int(gstep / len(self.dl_trn))
		cond = ep % freq == 0
		if not cond:
			return None
		figs = {}
		if self.model.cfg.dec_type == 'lin':
			# w_dec
			div = 8 if self.model.cfg.dataset == 'BALLS' else 32
			if self.model.cfg.type == 'categorical':
				nrows = np.ceil(np.prod(self.model.size) / div)
			else:
				nrows = np.ceil(self.model.cfg.n_latents / div)
			fig, _ = self.model.show(
				which='dec',
				order=order,
				nrows=int(nrows),
				add_title=False,
				display=False,
				dpi=80,
			)
			figs['w_dec'] = fig
		else:
			# recon & samples
			pad = 0 if self.model.cfg.dataset == 'MNIST' else 1
			kws = dict(t=temp, pad=pad, display=False)
			figs['recon'] = self.show_recon(**kws)[0]
			figs['samples'] = self.show_samples(**kws)[0]

		# Log dict of images
		log_dict = {}
		for name, f in figs.items():
			log_dict[f'figs/{name}'] = wandb.Image(f)
		wandb.log(log_dict, step=gstep)

		# Close figs to free memory
		for f in figs.values():
			plt.close(f)

		return figs

	def reset_model(self):
		mode_class = MODEL_CLASSES[self.model.cfg.type]
		self.model = mode_class(self.model.cfg).to(self.device)
		if self.model_ema is not None:
			self.model_ema = mode_class(self.model.cfg).to(self.device)
		return

	def _init_fun(self):
		def _fun(x):
			if self.model.cfg.type == 'poisson':
				if self.cfg.method == 'mc':
					dist, log_dr, z, y = self.model(x)
					recon_batch = self.model.loss_recon(y, x)
				elif self.cfg.method == 'exact':
					output = self.model.loss_recon_exact(x)
					recon_batch, dist, log_dr = output
				else:
					raise ValueError(self.cfg.method)
				kl = self.model.loss_kl(log_dr)

			elif self.model.cfg.type in [
					'gaussian', 'laplace', 'categorical']:
				if self.cfg.method == 'mc':
					dist, z, y = self.model(x)
					recon_batch = self.model.loss_recon(y, x)
				elif self.cfg.method == 'exact':
					output = self.model.loss_recon_exact(x)
					recon_batch, dist, _ = output
				else:
					raise ValueError(self.cfg.method)
				kl = self.model.loss_kl(dist)

			else:
				raise ValueError(self.model.cfg.type)

			return recon_batch, kl, dist

		self._fun = _fun
		return


def save_fit_info(
		args: dict,
		tr: TrainerVAE,
		start: str,
		stop: str = None,
		save_dir: str = 'logs',
		root: str = 'Dropbox/git/PoissonVAE', ):
	stop = stop or now(True)
	# make info string
	host = os.uname().nodename
	done = f"[PROGRESS] fitting VAE on {host}-cuda:{args['device']} done!"
	done += f" run time  ——>  {time_dff_string(start, stop)}  <——\n"
	done += f"[PROGRESS] start: {start}  ———  stop: {stop}\n"
	info = tr.info()
	info += f"\n\n{'_' * 100}\n[INFO] args:\n{json.dumps(args, indent=4)}"
	info = f"{done}\n{info}"

	# file name
	fname = [
		tr.model.cfg.model_str,
		args['archi'] if
		args['archi'].endswith('>')
		else f"<{args['archi']}>",
		tr.model.cfg.dataset,
		'_'.join([
			f"b-{tr.cfg.kl_beta:0.3g}",
			f"k-{tr.model.cfg.n_latents}"]),
		# tr.cfg.method,
	]
	if args.get('comment') is not None:
		fname += [args['comment']]
	fname = '-'.join(fname)
	fname = '_'.join([
		fname,
		f"{host}-{tr.model.cfg.seed}",
		f"({stop})",
	])
	# save dir
	save_dir = pjoin(add_home(root), save_dir)
	os.makedirs(save_dir, exist_ok=True)
	# save
	save_obj(
		obj=info,
		file_name=fname,
		save_dir=save_dir,
		mode='txt',
	)
	return


def _setup_args() -> argparse.Namespace:
	from main.config_vae import (
		DATA_CHOICES,
		METHOD_CHOICES,
		T_ANNEAL_CHOICES,
	)
	parser = argparse.ArgumentParser()

	#####################
	# setup
	#####################
	parser.add_argument(
		"device",
		choices=range(torch.cuda.device_count()),
		help='cuda:device',
		type=int,
	)
	parser.add_argument(
		"dataset",
		choices=DATA_CHOICES,
		type=str,
	)
	parser.add_argument(
		"model",
		choices=MODEL_CLASSES,
		type=str,
	)
	parser.add_argument(
		"archi",
		help='architecture type',
		type=str,
	)
	########################
	# main cfgs (common)
	########################
	parser.add_argument(
		"--n_latents",
		help='# latents',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--n_ch",
		help='# channels',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--init_dist",
		help='init dist (fc_enc)',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, str),
	)
	parser.add_argument(
		"--init_scale",
		help='init scale (fc_enc)',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--activation_fn",
		help='activation function',
		default='swish',
		type=str,
	)
	parser.add_argument(
		"--fit_prior",
		help='fit prior?',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, true_fn),
	)
	parser.add_argument(
		"--enc_norm",
		help='weight norm (fc_enc)',
		default=False,
		type=true_fn,
	)
	parser.add_argument(
		"--dec_norm",
		help='weight norm (fc_dec)',
		default=False,
		type=true_fn,
	)
	parser.add_argument(
		"--use_bn",
		help='use batch norm?',
		default=False,
		type=true_fn,
	)
	parser.add_argument(
		"--use_se",
		help='use squeeze & excite?',
		default=True,
		type=true_fn,
	)
	parser.add_argument(
		"--seed",
		help='random seed',
		default=0,
		type=int,
	)
	########################
	# main cfgs (specific)
	########################
	# poisson
	parser.add_argument(
		"--prior_clamp",
		help='prior init clamp',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--prior_log_dist",
		help='prior dist type',
		default='uniform',
		type=str,
	)
	parser.add_argument(
		"--indicator_approx",
		help='soft indicator function',
		default='sigmoid',
		type=str,
	)
	parser.add_argument(
		"--exc_only",
		help='only excitation?',
		default=False,
		type=true_fn,
	)
	# categorical
	parser.add_argument(
		"--n_categories",
		help='# categories',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	# gaussian & laplace
	parser.add_argument(
		"--latent_act",
		help='activation on z?',
		default=None,
		type=str,
	)
	########################
	# trainer cfgs
	########################
	parser.add_argument(
		"--method",
		choices=METHOD_CHOICES,
		help='training method',
		default='mc',
		type=str,
	)
	parser.add_argument(
		"--lr",
		help='learning rate',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--epochs",
		help='# epochs',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--batch_size",
		help='batch size',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, int),
	)
	parser.add_argument(
		"--warm_restart",
		help='# warm restarts',
		default=0,
		type=int,
	)
	parser.add_argument(
		"--warmup_portion",
		help='warmup portion',
		default=0.01,
		type=float,
	)
	parser.add_argument(
		"--optimizer",
		help='optimizer',
		default='adamax_fast',
		type=str,
	)
	# temp
	parser.add_argument(
		"--temp_start",
		help='temp: [start] —> stop',
		default=1.0,
		type=float,
	)
	parser.add_argument(
		"--temp_stop",
		help='temp: start —> [stop]',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--temp_anneal_type",
		choices=T_ANNEAL_CHOICES,
		help='temp anneal type',
		default='lin',
		type=str,
	)
	parser.add_argument(
		"--temp_anneal_portion",
		help='temp anneal portion',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	# kl
	parser.add_argument(
		"--kl_beta",
		help='kl loss beta coefficient',
		default=1.0,
		type=float,
	)
	parser.add_argument(
		"--kl_anneal_portion",
		help='kl beta anneal portion',
		default=0.5,
		type=float,
	)
	parser.add_argument(
		"--kl_const_portion",
		help='kl const portion',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--kl_anneal_cycles",
		help='0: linear, >0: cosine',
		default=0,
		type=int,
	)
	# lamb
	parser.add_argument(
		"--lambda_anneal",
		help='anneal weight reg coeff?',
		default=False,
		type=true_fn,
	)
	parser.add_argument(
		"--lambda_norm",
		help='weight regularization strength',
		default=0.0,
		type=float,
	)
	parser.add_argument(
		"--grad_clip",
		help='gradient norm clipping',
		default='__placeholder__',
		type=lambda v: placeholder_fn(v, float),
	)
	parser.add_argument(
		"--chkpt_freq",
		help='checkpoint freq',
		default=50,
		type=int,
	)
	parser.add_argument(
		"--eval_freq",
		help='eval freq',
		default=20,
		type=int,
	)
	parser.add_argument(
		"--log_freq",
		help='log freq',
		default=10,
		type=int,
	)
	parser.add_argument(
		"--comment",
		help='comment',
		default=None,
		type=str,
	)
	parser.add_argument(
		"--use_amp",
		help='automatic mixed precision?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--dry_run",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--cudnn_bench",
		help='use cudnn benchmark?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--verbose",
		help='to make sure config is alright',
		action='store_true',
		default=False,
	)
	########################
	# Wandb
	########################
	parser.add_argument(
		"--wandb_project",
		type=str,
		default='PoissonVAE',
		help="Wandb project name",
	)
	parser.add_argument(
		"--wandb_entity",
		type=str,
		default=None,
		help="Wandb entity (username or team)",
	)
	parser.add_argument(
		"--no_wandb",
		action="store_true",
		default=False,
		help="Disable wandb logging",
	)

	return parser.parse_args()


def _main():
	args = _setup_args()
	cfg_vae, cfg_tr = default_configs(
		dataset=args.dataset,
		model_type=args.model,
		archi_type=args.archi,
	)
	# filter: removes unrelated kwargs
	cfg_vae = filter_kwargs(CFG_CLASSES[args.model], cfg_vae)
	cfg_tr = filter_kwargs(ConfigTrainVAE, cfg_tr)
	# convert from dict to cfg objects, back to dict
	# this step adds the full set of default values
	cfg_vae['save'] = False
	cfg_vae = obj_attrs(CFG_CLASSES[args.model](**cfg_vae))
	cfg_tr = obj_attrs(ConfigTrainVAE(**cfg_tr))
	# override with values provided by user
	for k, v in vars(args).items():
		if v == '__placeholder__':
			continue
		if k in cfg_vae:
			cfg_vae[k] = v
		if k in cfg_tr:
			cfg_tr[k] = v
	# final convert: from dict to cfg objects
	cfg_vae['save'] = not args.dry_run
	cfg_vae = CFG_CLASSES[args.model](**cfg_vae)
	cfg_tr = ConfigTrainVAE(**cfg_tr)

	# custom modifications
	if cfg_tr.method == 'exact' and 'mlp' in args.archi:
		cfg_tr.grad_clip *= 4

	# manually inject WandB args into cfg_tr
	cfg_tr.wandb_project = args.wandb_project
	cfg_tr.wandb_entity = args.wandb_entity
	cfg_tr.no_wandb = args.no_wandb

	# main & tr
	device = f"cuda:{args.device}"
	vae = MODEL_CLASSES[args.model](cfg_vae)
	tr = TrainerVAE(
		model=vae,
		cfg=cfg_tr,
		device=device,
		verbose=args.verbose,
	)

	if args.verbose:
		print(args)
		# vae.print()
		msg = '\n'.join([
			f"\ntotal num iters: {tr.n_iters}",
			f"{vae.cfg.name()}",
			f"{tr.cfg.name()}_({vae.timestamp})\n",
		])
		print(msg)

	if args.comment is not None:
		comment = '_'.join([
			args.comment,
			tr.cfg.name(),
		])
	else:
		comment = tr.cfg.name()

	if args.cudnn_bench:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.benchmark_limit = 0

	start = now(True)
	if not args.dry_run:
		tr.train(comment)
		save_fit_info(vars(args), tr, start)
	return


if __name__ == "__main__":
	_main()
