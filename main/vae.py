from base.common import *
from base.distributions import (
	dists, softclamp, softclamp_upper,
	Normal, Laplace, Poisson, Categorical,
)
from figures.imgs import plot_weights


class BaseVAE(Module):
	def __init__(self, cfg: ConfigVAE, **kwargs):
		super(BaseVAE, self).__init__(cfg, **kwargs)
		self.Dist: dists.Distribution()
		self.register_buffer(
			name='temp',
			tensor=torch.tensor(1.0),
		)
		self._init()

	def forward(self, x):
		raise NotImplementedError

	def infer(self, x):
		raise NotImplementedError

	@torch.no_grad()
	def xtract_ftr(self, x, t: float = 0.0):
		raise NotImplementedError

	@torch.no_grad()
	def sample(self, n: int, t: float = 0.0):
		raise NotImplementedError

	def update_t(self, new_t: float):
		if new_t is None:
			return
		assert new_t >= 0.0, "must be non-neg"
		self.temp.fill_(new_t)
		return

	def encode(self, x):
		if self.cfg.enc_type == 'conv':
			x = self.stem(x)
			x = self.enc(x)
		elif self.cfg.enc_type == 'lin':
			x = x.flatten(start_dim=1)
		elif self.cfg.enc_type == 'mlp':
			x = x.flatten(start_dim=1)
			x = self.enc(x)
		else:
			raise ValueError(self.cfg.enc_type)
		x = self.fc_enc(x)
		return x

	def decode(self, z):
		if self.cfg.dec_type == 'conv':
			h = self.fc_dec(z)
			h = h.view(*self.shape)
			x_recon = self.dec(h)
		elif self.cfg.dec_type == 'lin':
			x_recon = self.fc_dec(z).view(*self.shape)
		elif self.cfg.dec_type == 'mlp':
			h = self.fc_dec(z)
			x_recon = self.dec(h).view(*self.shape)
		else:
			raise ValueError(self.cfg.enc_type)
		return x_recon

	def loss_recon(self, y, x):
		return torch.sum(
			self.mse(y, x),
			dim=[1, 2, 3],
		)

	def loss_recon_exact(self, x):
		assert self.cfg.dec_type == 'lin', \
			"only valid for linear decoder"
		# infer -> posterior dist
		output = self.infer(x)
		if isinstance(output, tuple):
			dist, etc = output
		else:
			dist, etc = output, None
		# mean & variance
		mu = dist.mean.flatten(start_dim=1)
		var = dist.variance.flatten(start_dim=1)
		# decoder weights
		phi = self.fc_dec.get_weight()
		gram_mat = phi.pow(2).sum(0)

		mse = x.flatten(start_dim=1) - mu @ phi.T
		mse = mse.pow(2).sum(1)
		recon_batch = mse + var @ gram_mat
		return recon_batch, dist, etc

	def loss_kl(self, *args, **kwargs):
		raise NotImplementedError

	def loss_weight(self):
		if not self.all_lognorm:
			return None
		return torch.cat(self.all_lognorm).pow(2).mean()

	def find_dead_neurons(self, frac: int = 8):
		norms = tonp(torch.linalg.vector_norm(
			self.fc_dec.weight, dim=0))
		eps = np.finfo(norms.dtype).eps
		norms = np.maximum(norms, eps)
		log_norms = np.log(norms)

		# finds large contigouous gap
		bins = np.linspace(
			start=np.nanmin(log_norms),
			stop=np.nanmax(log_norms),
			num=len(log_norms) * 10
		)
		hist, _ = np.histogram(
			log_norms, bins=bins)
		median_idx = int(np.digitize(
			x=np.median(log_norms),
			bins=bins,
		)) - 1
		idx = find_last_contiguous_zeros(
			mask=hist[:median_idx] > 0,
			w=len(log_norms) * 2,
		)
		dead1 = log_norms < bins[idx]

		# finds smallest and largest
		bins = np.linspace(
			start=np.nanmin(log_norms),
			stop=np.nanmax(log_norms),
			num=len(log_norms) // frac
		)
		hist, _ = np.histogram(
			log_norms, bins=bins)
		i, j = find_critical_ids(hist > 0)
		dead2 = np.logical_or(
			log_norms < bins[i],
			log_norms > bins[j],
		)
		return dead1 | dead2

	def show(
			self,
			which: str = 'dec',
			order: Iterable[int] = None,
			method: str = 'min-max',
			add_title: bool = False,
			display: bool = True,
			**kwargs, ):
		assert which in ['enc', 'dec']
		# get weights
		w = f"fc_{which}"
		w = getattr(self, w)
		w = w.get_weight().data
		if which == 'dec':
			w = w.T
		w = tonp(w.reshape(self.shape))
		if order is not None:
			w = w[order]

		fig, ax = plot_weights(
			w=w,
			method=method,
			title=None if not add_title else
			which.capitalize() + 'oder',
			display=display,
			**kwargs,
		)
		return fig, ax

	def _init(self):
		self.mse = nn.MSELoss(reduction='none')
		self._init_kws()
		self._init_enc()
		self._init_dec()
		self._init_norm()
		self._init_prior()
		self._init_weight()
		return

	def _init_kws(self):
		is_conv = (
			self.cfg.enc_type == 'conv' or
			self.cfg.dec_type == 'conv'
		)
		if is_conv:
			self._kws_conv = dict(
				reg_lognorm=True,
				act_fn=self.cfg.activation_fn,
				use_bn=self.cfg.use_bn,
				use_se=self.cfg.use_se,
				eps=self.cfg.res_eps,
				n_nodes=0,
				scale=1.0,
			)
		else:
			self._kws_conv = {}
		return

	def _init_enc(self):
		normalize_dim = 1
		if self.cfg.enc_type == 'conv':
			# stem
			if self.cfg.dataset in ['vH16', 'CIFAR16', 'BALLS16', 'BALLS64']:
				padding = 1
			elif self.cfg.dataset.endswith('MNIST'):
				padding = 'valid'
			else:
				raise ValueError(self.cfg.dataset)
			kws = dict(
				in_channels=1,
				out_channels=self.cfg.n_ch,
				kernel_size=3,
				padding=padding,
				reg_lognorm=True,
			)
			self.stem = Conv2D(**kws)
			# sequential
			self._kws_conv['n_nodes'] = 2
			self.enc = _build_conv_enc(
				nch=self.cfg.n_ch,
				kws=self._kws_conv,
				dataset=self.cfg.dataset,
			)
			# final fc step
			self.fc_enc = Linear(
				in_features=self.enc[-1].dim,
				out_features=self._enc_out_channel(),
				normalize=self.cfg.enc_norm,
				normalize_dim=normalize_dim,
				bias=self.cfg.enc_bias,
			)

		elif self.cfg.enc_type == 'mlp':
			self.enc = ResDenseLayer(
				dim=self.cfg.input_sz ** 2,
				expand=self.cfg.n_ch,
			)
			self.fc_enc = Linear(
				in_features=self.cfg.input_sz ** 2,
				out_features=self._enc_out_channel(),
				normalize=self.cfg.enc_norm,
				normalize_dim=normalize_dim,
				bias=self.cfg.enc_bias,
			)

		elif self.cfg.enc_type == 'lin':
			self.fc_enc = Linear(
				in_features=self.cfg.input_sz ** 2,
				out_features=self._enc_out_channel(),
				normalize=self.cfg.enc_norm,
				normalize_dim=normalize_dim,
				bias=self.cfg.enc_bias,
			)

		else:
			raise ValueError(self.cfg.enc_type)

		return

	def _init_dec(self):
		normalize_dim = 0
		if self.cfg.dec_type == 'conv':
			spat_dim = 2
			nch = self.cfg.n_ch * 8
			self.shape = (-1, nch, spat_dim, spat_dim)
			self.fc_dec = Linear(
				in_features=self._dec_in_channel(),
				out_features=nch * spat_dim ** 2,
				normalize=self.cfg.dec_norm,
				normalize_dim=normalize_dim,
				bias=self.cfg.dec_bias,
			)
			# sequential
			self._kws_conv['n_nodes'] = 1
			self.dec = _build_conv_dec(
				nch=nch,
				kws=self._kws_conv,
				dataset=self.cfg.dataset,
			)

		elif self.cfg.dec_type == 'mlp':
			shape = (self.cfg.input_sz,) * 2
			self.shape = (-1, 1, *shape)
			self.fc_dec = Linear(
				in_features=self._dec_in_channel(),
				out_features=self.cfg.input_sz ** 2,
				normalize=self.cfg.dec_norm,
				normalize_dim=normalize_dim,
				bias=self.cfg.dec_bias,
			)
			self.dec = nn.Sequential(
				get_act_fn(self.cfg.activation_fn),
				ResDenseLayer(self.cfg.input_sz ** 2),
				get_act_fn(self.cfg.activation_fn),
				nn.Linear(
					self.cfg.input_sz ** 2,
					self.cfg.input_sz ** 2,
					bias=True),
			)

		elif self.cfg.dec_type == 'lin':
			shape = (self.cfg.input_sz,) * 2
			self.shape = (-1, 1, *shape)
			self.fc_dec = Linear(
				in_features=self._dec_in_channel(),
				out_features=self.cfg.input_sz ** 2,
				normalize=self.cfg.dec_norm,
				normalize_dim=normalize_dim,
				bias=self.cfg.dec_bias,
			)

		else:
			raise ValueError(self.cfg.dec_type)

		return

	def _init_norm(self, regul_list: List[str] = None):
		regul_list = regul_list if regul_list else [
			'enc', 'dec']
		self.all_lognorm = []
		for child_name, child in self.named_children():
			for m in child.modules():
				cond = (
					isinstance(m, (Conv2D, DeConv2D))
					and child_name in regul_list
					and m.lognorm.requires_grad
				)
				if cond:
					self.all_lognorm.append(m.lognorm)
		return

	def _init_weight(self):
		if self.cfg.init_scale is None:
			return
		kws = dict(
			dist=self.cfg.init_dist,
			scale=self.cfg.init_scale,
			loc=0,
		)
		init = Initializer(**kws)
		init.apply(self.fc_enc.weight)
		if self.cfg.dec_type == 'lin':
			init.apply(self.fc_dec.weight)
		if self.fc_enc.bias is not None:
			nn.init.zeros_(self.fc_enc.bias)
		return

	def _init_prior(self):
		raise NotImplementedError

	def _enc_out_channel(self):
		if self.cfg.type == 'poisson':
			co = self.cfg.n_latents
		elif self.cfg.type in ['gaussian', 'laplace']:
			co = self.cfg.n_latents * 2
		elif self.cfg.type == 'categorical':
			co = np.prod(self.size)
		else:
			raise ValueError(self.cfg.type)
		return co

	def _dec_in_channel(self):
		if self.cfg.type in ['poisson', 'gaussian', 'laplace']:
			ci = self.cfg.n_latents
		elif self.cfg.type == 'categorical':
			ci = np.prod(self.size)
		else:
			raise ValueError(self.cfg.type)
		return ci


class PoissonVAE(BaseVAE):
	def __init__(self, cfg: ConfigPoisVAE, **kwargs):
		super(PoissonVAE, self).__init__(cfg, **kwargs)
		self.Dist = Poisson
		self.register_buffer(
			name='n_exp',
			tensor=torch.tensor(0),
		)
		self.update_n(200.0)

	def forward(self, x):
		# infer
		dist, log_dr = self.infer(x)
		# sample
		spks = dist.rsample()
		# decode
		y = self.decode(spks)
		return dist, log_dr, spks, y

	def infer(
			self,
			x: torch.Tensor,
			t: float = None,
			ablate: Sequence[int] = None, ):
		if t is None:
			t = self.temp
		log_r = self.log_rate.expand(len(x), -1)
		log_dr = self.encode(x)
		if self.cfg.exc_only:
			log_dr = softclamp(log_dr, 10.0, 0)
		else:
			log_dr = softclamp_upper(log_dr, 10.0)
		if ablate is not None:
			log_dr[:, ablate] = 0.0
		dist = self.Dist(
			log_rate=softclamp_upper(log_r + log_dr, 5.0),
			indicator_approx=self.cfg.indicator_approx,
			n_exp=self.n_exp,
			n_exp_p=1e-3,
			temp=t,
		)
		return dist, log_dr

	@torch.no_grad()
	def xtract_ftr(
			self,
			x: torch.Tensor,
			t: float = 0.0,
			ablate_enc: Sequence[int] = None,
			ablate_spks: Sequence[int] = None, ):
		dist, log_dr = self.infer(x, t, ablate_enc)
		spks = dist.rsample()
		if ablate_spks is not None:
			spks[:, ablate_spks] = 0.0
		y = self.decode(spks)
		return dist, log_dr, spks, y

	@torch.no_grad()
	def sample(self, n: int, t: float = 0.0):
		if t is None:
			t = self.temp
		log_r = self.log_rate.expand(n, -1)
		dist = self.Dist(
			log_rate=log_r,
			n_exp=self.n_exp,
			temp=t,
			indicator_approx=self.cfg.indicator_approx,
		)
		spks = dist.rsample()
		x_samples = self.decode(spks)
		return x_samples, spks

	def loss_kl(self, log_dr):
		log_r = self.log_rate.expand(len(log_dr), -1)
		f = 1 + torch.exp(log_dr) * (log_dr - 1)
		kl = torch.exp(log_r) * f
		return kl

	def update_n(self, rate: float):
		if rate is None:
			return
		assert rate > 0.0, "must be positive"
		dist = sp_stats.poisson(rate)
		n_exp = dist.ppf(1.0 - 1e-5)
		self.n_exp.fill_(int(n_exp))
		return

	def _init_prior(self):
		rng = get_rng(self.cfg.seed)
		kws = {'size': (1, self.cfg.n_latents)}
		if self.cfg.prior_log_dist == 'cte':
			log_rate = np.ones(kws['size'])
			log_rate *= self.cfg.prior_clamp
		elif self.cfg.prior_log_dist == 'uniform':
			kws.update(dict(
				low=-6.0,
				high=self.cfg.prior_clamp,
			))
			log_rate = rng.uniform(**kws)
		elif self.cfg.prior_log_dist == 'normal':
			s = np.abs(np.log(np.abs(
				self.cfg.prior_clamp)))
			kws.update(dict(loc=0.0, scale=s))
			log_rate = rng.normal(**kws)
		else:
			raise NotImplementedError(
				self.cfg.prior_log_dist)

		log_rate = torch.tensor(
			data=log_rate,
			dtype=torch.float,
		)
		log_rate[log_rate > 6.0] = 0.0

		self.log_rate = nn.Parameter(
			data=log_rate,
			requires_grad=True
		)
		return


class ContinuousVAE(BaseVAE):
	def __init__(self, cfg: Union[ConfigGausVAE, ConfigLapVAE], **kwargs):
		super(ContinuousVAE, self).__init__(cfg, **kwargs)

	def forward(self, x):
		# infer
		dist = self.infer(x)
		# sample
		z = dist.rsample()
		# decode
		z = self._act_fn(z)
		y = self.decode(z)
		return dist, z, y

	def infer(self, x, t: float = None):
		if t is None:
			t = self.temp
		h = self.encode(x)
		loc, log_scale = torch.chunk(h, 2, dim=1)
		dist = self._q(len(x), loc, log_scale, t)
		return dist

	@torch.no_grad()
	def xtract_ftr(self, x, t: float = 0.0):
		dist = self.infer(x, t)
		z = dist.sample()
		z = self._act_fn(z)
		y = self.decode(z)
		return dist, z, y

	@torch.no_grad()
	def sample(self, n: int, t: float = 1.0):
		if t is None:
			t = self.temp
		dist = self._q(n, t=t)
		z = dist.sample()
		z = self._act_fn(z)
		x_samples = self.decode(z)
		return x_samples, z

	def loss_kl(self, q):
		size = (len(q.loc), -1)
		p = self.Dist(
			loc=self.loc.expand(size),
			log_scale=self.log_scale.expand(size),
			temp=q.t,
		)
		return q.kl(p)

	def _q(
			self,
			n: int,
			loc: torch.tensor = 0,
			log_scale: torch.tensor = 0,
			t: float = 1.0, ):
		dist = self.Dist(
			loc=loc + self.loc.expand(n, -1),
			log_scale=log_scale + self.log_scale.expand(n, -1),
			temp=t,
		)
		return dist

	def _act_fn(self, z):
		act_fn = {
			'relu': F.relu,
			'softplus': F.softplus,
			'sigmoid': torch.sigmoid,
			'quartic': lambda x: x.pow(4),
			'square': torch.square,
			'exp': torch.exp,
		}.get(self.cfg.latent_act)
		if act_fn is not None:
			return act_fn(z)
		return z

	def _init_prior(self):
		size = (1, self.cfg.n_latents)
		self.loc = nn.Parameter(
			data=torch.zeros(size),
			requires_grad=self.cfg.fit_prior,
		)
		self.log_scale = nn.Parameter(
			data=torch.zeros(size),
			requires_grad=self.cfg.fit_prior,
		)
		return


class GaussianVAE(ContinuousVAE):
	def __init__(self, cfg: ConfigGausVAE, **kwargs):
		super(GaussianVAE, self).__init__(cfg, **kwargs)
		self.Dist = Normal


class LaplaceVAE(ContinuousVAE):
	def __init__(self, cfg: ConfigLapVAE, **kwargs):
		super(LaplaceVAE, self).__init__(cfg, **kwargs)
		self.Dist = Laplace


class CategoricalVAE(BaseVAE):
	def __init__(self, cfg: ConfigCatVAE, **kwargs):
		self.size = (cfg.n_latents, cfg.n_categories)
		super(CategoricalVAE, self).__init__(cfg, **kwargs)
		self.Dist = Categorical

	def forward(self, x):
		# infer
		dist = self.infer(x)
		# sample
		z = dist.rsample()
		z = z.flatten(start_dim=1)
		# decode
		y = self.decode(z)
		return dist, z, y

	def infer(self, x, t: float = None):
		if t is None:
			t = self.temp
		logits = self.encode(x)
		logits = logits.view(-1, *self.size)
		dist = self.Dist(logits=logits, temp=t)
		return dist

	@torch.no_grad()
	def xtract_ftr(self, x, t: float = 0.0):
		dist = self.infer(x, t)
		z = dist.sample()
		z = z.flatten(start_dim=1)
		y = self.decode(z)
		return dist, z, y

	@torch.no_grad()
	def sample(self, n: int, t: float = 0.0):
		if t is None:
			t = self.temp
		logits = self.logits.expand(n, -1, -1)
		dist = self.Dist(logits=logits, temp=t)
		z = dist.sample().flatten(start_dim=1)
		x_samples = self.decode(z)
		return x_samples, z

	def loss_kl(self, q):
		size = (len(q.logits), -1, -1)
		logits = self.logits.expand(size)
		p = dists.Categorical(logits=logits)
		return q.kl(p)

	def _init_prior(self):
		fill_val = 1.0 / self.cfg.n_categories
		probs = torch.full(
			size=(1, *self.size),
			fill_value=fill_val,
		)
		eps = torch.finfo(torch.float).eps
		self.logits = nn.Parameter(
			data=torch.logit(probs, eps=eps),
			requires_grad=self.cfg.fit_prior,
		)
		return


class ResDenseLayer(nn.Module):
	def __init__(
			self,
			dim: int,
			expand: int = 8,
			drop: float = 0.1,
	):
		super(ResDenseLayer, self).__init__()
		self.fc1 = nn.Linear(dim, dim * expand)
		self.fc2 = nn.Linear(dim * expand, dim)
		self.layer_norm = nn.LayerNorm(dim)
		self.drop = nn.Dropout(drop)
		self.relu = nn.ReLU()
		self.dim = dim

	def forward(self, x):
		skip = x
		x = self.fc1(x)
		x = self.relu(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.layer_norm(x + skip)
		return x


class ResConvPool(nn.Module):
	def __init__(
			self,
			dim: int,
			act_fn: str = 'none',
			eps: float = 1.0,
			**kwargs, ):
		super(ResConvPool, self).__init__()
		defaults = dict(
			kernel_size=4,
			padding='valid',
			reg_lognorm=True,
		)
		kwargs = setup_kwargs(defaults, kwargs)
		self.act_fn = get_act_fn(act_fn, False)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.conv = Conv2D(dim, dim, **kwargs)
		self.eps = eps

	def forward(self, x):
		skip = self.pool(x)
		if self.act_fn is not None:
			x = self.act_fn(x)
		x = self.conv(x)
		return skip + self.eps * x


# noinspection PyTypeChecker
def _build_conv_enc(
		nch: int,
		kws: dict,
		dataset: str,
		add_fc: bool = False, ) -> nn.Sequential:

	if dataset.endswith('MNIST'):
		kws['factorized'] = False  # FactorizedReduce incompatible
		n_layers = 3
	elif dataset in ['vH16', 'CIFAR16']:
		n_layers = 2
	elif dataset.startswith('BALLS'):
		n_layers = 4
	else:
		raise ValueError(dataset)

	layers = [Cell(nch, nch, 'normal_pre', **kws)]
	# conv
	for _ in range(n_layers):
		layers.extend([
			Cell(nch, nch * MULT, 'down_enc', **kws),
			Cell(nch * MULT, nch * MULT, 'normal_pre', **kws),
		])
		nch *= MULT
	# pool + flatten
	kws_conv_pool = dict(
		dim=nch,
		kernel_size=4,
		padding='valid',
		reg_lognorm=kws['reg_lognorm'],
		act_fn=kws['act_fn'],
		eps=kws['eps'],
	)
	layers.extend([
		ResConvPool(**kws_conv_pool),
		nn.Flatten(start_dim=1),
	])
	# fc?
	if add_fc:
		for _ in range(n_layers):
			layers.extend([
				get_act_fn(kws['act_fn'], True),
				nn.Linear(nch, nch // MULT, bias=True),
			])
			nch //= MULT

	return nn.Sequential(*layers, ResDenseLayer(nch))


# noinspection PyTypeChecker
def _build_conv_dec(
		nch: int,
		kws: dict,
		dataset: str, ) -> nn.Sequential:

	if dataset.endswith('MNIST'):
		n_layers = 4
	elif dataset in ['vH16', 'CIFAR16', 'BALLS']:
		n_layers = 3
	else:
		raise ValueError(dataset)

	layers = [Cell(nch, nch, 'normal_dec', **kws)]
	for i in range(n_layers):
		layers.extend([
			Cell(nch, nch // MULT, 'up_dec', **kws),
			Cell(nch // MULT, nch // MULT, 'normal_dec', **kws),
		])
		nch //= MULT
		if dataset.endswith('MNIST') and i == 2:
			layers.append(nn.Upsample(size=14))

	kws_final = dict(
		in_channels=nch,
		out_channels=1,
		kernel_size=1,
		padding='valid',
		reg_lognorm=True,
	)
	return nn.Sequential(*layers, Conv2D(**kws_final))


def _build_mlp(n_dims: int, n_layers: int = 3):
	net = []
	nch = n_dims
	for _ in range(n_layers):
		net.extend([
			nn.SiLU(inplace=True),
			nn.Linear(nch, nch // MULT, bias=True),
			nn.SiLU(inplace=True),
			nn.Linear(nch // MULT, nch // MULT, bias=True),
		])
		nch //= MULT
	return nn.Sequential(*net)


MODEL_CLASSES = {
	'poisson': PoissonVAE,
	'gaussian': GaussianVAE,
	'laplace': LaplaceVAE,
	'categorical': CategoricalVAE,
}
