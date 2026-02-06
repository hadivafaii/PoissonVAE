from base.config_base import *

_LOG_DIST_CHOICES = ['cte', 'uniform', 'normal']
_ARCHI_CHOICES = ['lin', 'conv', 'mlp']
T_ANNEAL_CHOICES = ['lin', 'exp']
METHOD_CHOICES = ['mc', 'exact']
DATA_CHOICES = [
	'vH16', 'CIFAR16',
	'MNIST', 'EMNIST',
	'FashionMNIST', 'CIFAR10',
]


class ConfigVAE(BaseConfig):
	@validate_choices(DATA_CHOICES, 'dataset')
	@validate_choices(_ARCHI_CHOICES, 'enc_type')
	@validate_choices(_ARCHI_CHOICES, 'dec_type')
	def __init__(
			self,
			dataset: str,
			n_ch: int = 32,
			n_latents: int = 512,
			enc_type: str = 'lin',
			dec_type: str = 'lin',
			enc_bias: bool = False,
			dec_bias: bool = False,
			enc_norm: bool = False,
			dec_norm: bool = False,
			fit_prior: bool = False,
			activation_fn: str = 'swish',
			init_dist: str = 'Normal',
			init_scale: float = 0.05,
			res_eps: float = 1.0,
			use_bn: bool = False,
			use_se: bool = True,
			**kwargs,
	):
		"""
		:param dataset: {
			'vH16', 'CIFAR16',  # amortized sparse coding
			'MNIST', 'CIFAR10',  # representation learning
			'BALLS',  # extrapolation
		}
		:param n_ch:
			# type = conv: used in the conventional way (model width)
			# type = mlp: determines 'expand' ratio in ResDenseLayer
			# type = lin: not used —— removed from name()
		:param n_latents: dimensionality of the latent space
		:param enc_type: {'lin', 'conv', 'mlp'}
		:param dec_type: {'lin', 'conv', 'mlp'}
		:param enc_bias: {True, False}
		:param dec_bias: {True, False}
		:param enc_norm: {True, False}
		:param dec_norm: {True, False}
		:param activation_fn: {'swish', 'elu', 'relu', 'leaky_relu'}
		:param init_dist: distribution useed to init 'fc_enc' weights
		:param init_scale: weight init scale, only affects 'fc_enc'
		:param res_eps: residual layer output: skip + eps * x
		:param use_bn: BatchNorm?
		:param use_se: Squeeze&Excite?
		:param kwargs:
		"""

		self.enc_type = enc_type
		self.dec_type = dec_type
		self.enc_bias = enc_bias
		self.dec_bias = dec_bias
		self.enc_norm = enc_norm
		self.dec_norm = dec_norm
		self.dataset = dataset
		self._init_input_sz()

		self.n_ch = n_ch
		self.n_latents = n_latents
		self.fit_prior = fit_prior

		self.use_bn = use_bn
		self.use_se = use_se
		self.res_eps = res_eps
		self.init_dist = init_dist
		self.init_scale = init_scale
		self.activation_fn = activation_fn

		super(ConfigVAE, self).__init__(**kwargs)

	@property
	def type(self):
		return NotImplemented

	@property
	def model_str(self):
		latent_name = getattr(
			self, 'latent_act', None)
		if latent_name is None:
			return self.type
		return '+'.join([self.type, latent_name])

	def _init_input_sz(self):
		dim_is_in_name = (
			self.dataset.startswith('BALLS') or
			self.dataset in ['vH16', 'CIFAR16']
		)
		if dim_is_in_name:
			self.input_sz = int_from_str(self.dataset)
		elif self.dataset == 'BALLS':
			self.input_sz = 16
		elif self.dataset.endswith('MNIST'):
			self.input_sz = 28
		else:
			raise ValueError(self.dataset)
		return

	def _special_name(self):
		raise NotImplementedError

	def _name(self):
		name = [
			self._special_name(),
			str(self.dataset),
			f"z-{self.n_latents}",
		]
		# n_ch irrelevant for linear model
		cond_nonlin = (
			self.enc_type != 'lin' or
			self.dec_type != 'lin'
		)
		if cond_nonlin:
			name += [f"k-{self.n_ch}"]
		# prior is fit?
		if self.fit_prior:
			name += ['fp']
		# weight norm?
		name += [
			'nrm-enc-dec' if self.enc_norm and self.dec_norm else
			'nrm-enc' if self.enc_norm else
			'nrm-dec' if self.dec_norm else
			'nrm-none',
		]
		if self.res_eps != 1.0:
			name += [f"eps-{self.res_eps:0.1g}"]
		# add achitecture specs
		archi = self.attr2archi()
		name = '_'.join(name + [archi])

		return name

	def attr2archi(self) -> str:
		enc_str = ''.join([
			f"<{self.enc_type}",
			'+b' if self.enc_bias else '',
		])
		dec_str = ''.join([
			str(self.dec_type),
			'+b>' if self.dec_bias else '>',
		])
		archi = '|'.join([
			enc_str,
			dec_str,
		])
		return archi

	def save(self):
		kws = dict(
			with_base=True,
			overwrite=True,
			verbose=False,
		)
		self._save(**kws)


class ConfigPoisVAE(ConfigVAE):
	def __init__(
			self,
			prior_clamp: float = -2.0,
			prior_log_dist: str = 'uniform',
			indicator_approx: str = 'sigmoid',
			exc_only: bool = False,
			**kwargs,
	):
		assert prior_log_dist in _LOG_DIST_CHOICES, \
			f"allowed prior log_dists:\n{_LOG_DIST_CHOICES}"
		self.indicator_approx = indicator_approx
		self.prior_log_dist = prior_log_dist
		self.prior_clamp = prior_clamp
		self.exc_only = exc_only
		super(ConfigPoisVAE, self).__init__(
			**kwargs)

	@property
	def type(self):
		return 'poisson'

	def _special_name(self):
		special_name = [
			str(self.type),
			str(self.prior_log_dist),
			f"c({self.prior_clamp:0.2g})",
		]
		if self.exc_only:
			special_name += ['exc']
		return '_'.join(special_name)


class ConfigCatVAE(ConfigVAE):
	def __init__(
			self,
			n_categories: int = 10,
			**kwargs,
	):
		self.n_categories = n_categories
		super(ConfigCatVAE, self).__init__(
			**kwargs)

	@property
	def type(self):
		return 'categorical'

	def _special_name(self):
		return '-'.join([
			str(self.type),
			str(self.n_categories),
		])


class ConfigContVAE(ConfigVAE):
	def __init__(
			self,
			model_type: str,
			latent_act: str = None,
			**kwargs,
	):
		self._type = model_type
		self.latent_act = latent_act
		super(ConfigContVAE, self).__init__(
			**kwargs)

	@property
	def type(self):
		return self._type

	def _special_name(self):
		special_name = [
			str(self.type),
		]
		if self.latent_act is not None:
			special_name += [
				str(self.latent_act),
			]
		return '_'.join(special_name)


class ConfigGausVAE(ConfigContVAE):
	def __init__(self, **kwargs):
		super(ConfigGausVAE, self).__init__(
			model_type='gaussian', **kwargs)


class ConfigLapVAE(ConfigContVAE):
	def __init__(self, **kwargs):
		super(ConfigLapVAE, self).__init__(
			model_type='laplace', **kwargs)


class ConfigTrainVAE(BaseConfigTrain):
	@validate_choices(METHOD_CHOICES, 'method')
	@validate_choices(T_ANNEAL_CHOICES, 'temp_anneal_type')
	def __init__(
			self,
			method: str = 'mc',
			kl_beta: float = 1.0,
			kl_beta_min: float = 1e-4,
			kl_anneal_cycles: int = 0,
			kl_anneal_portion: float = 0.5,
			kl_const_portion: float = 1e-2,
			lambda_anneal: bool = False,
			lambda_init: float = 0.0,
			lambda_norm: float = 0.0,
			temp_anneal_portion: float = 0.5,
			temp_anneal_type: str = 'lin',
			temp_start: float = 1.0,
			temp_stop: float = 0.05,
			**kwargs,
	):
		"""
		:param method: {'mc', 'exact'}
		:param kl_beta:
		:param kl_beta_min:
		:param kl_anneal_cycles:
		:param kl_anneal_portion:
		:param kl_const_portion:
		:param lambda_anneal:
		:param lambda_init:
		:param lambda_norm:
		:param temp_anneal_portion:
		:param temp_anneal_type:
		:param temp_start:
		:param temp_stop:
		:param kwargs:
		"""
		defaults = dict(
			lr=0.002,
			epochs=1200,
			batch_size=200,
			warm_restart=0,
			warmup_portion=0.01,
			optimizer='adamax_fast',
			scheduler_type='cosine',
			grad_clip=500,
			chkpt_freq=50,
			eval_freq=20,
			log_freq=10,
		)
		kwargs = setup_kwargs(defaults, kwargs)
		super(ConfigTrainVAE, self).__init__(**kwargs)
		self.set_optim_kws()  # reset optim kws
		self.set_scheduler_kws()  # reset scheduler kws
		assert 0.0 <= kl_anneal_portion <= 1.0
		assert 0.0 <= temp_anneal_portion <= 1.0

		self.method = method

		self.kl_beta = kl_beta
		self.kl_beta_min = kl_beta_min
		self.kl_anneal_cycles = kl_anneal_cycles
		self.kl_anneal_portion = kl_anneal_portion
		self.kl_const_portion = kl_const_portion
		self.lambda_anneal = lambda_anneal
		self.lambda_init = lambda_init
		self.lambda_norm = lambda_norm
		self.temp_anneal_portion = temp_anneal_portion
		self.temp_anneal_type = temp_anneal_type
		self.temp_start = temp_start
		self.temp_stop = temp_stop

	def name(self):
		name = [
			str(self.method),
			'-'.join([
				f"b{self.batch_size}",
				f"ep{self.epochs}",
				f"lr({self.lr:0.2g})",
			]),
			'-'.join([
				f"beta({self.kl_beta:0.2g}"
				f":{self.kl_anneal_cycles}"
				f"x{self.kl_anneal_portion:0.1g})",
			]),
		]
		if self.temp_anneal_portion > 0:
			temp_str = [
				'temp',
				f"({self.temp_stop:0.2g}:",
				'-'.join([
					f"{self.temp_anneal_type}",
					f"{self.temp_anneal_portion:0.1g})"]),
			]
			name.append(''.join(temp_str))
		if self.lambda_norm > 0:
			name.append(f"lamb({self.lambda_norm:0.2g})")
		if self.grad_clip is not None:
			name.append(f"gr({self.grad_clip})")
		return '_'.join(name)

	def save(self, save_dir: str):
		kws = dict(
			with_base=True,
			overwrite=True,
			verbose=False,
		)
		self._save(save_dir, **kws)


def default_configs(
		dataset: str,
		model_type: str,
		archi_type: str, ):

	########################
	# main —— model specific
	########################
	if model_type == 'poisson':
		cfg_vae = dict(
			prior_clamp=-2,
			fit_prior=True,
		)
		cfg_tr = dict()
	elif model_type in ['gaussian', 'laplace']:
		cfg_vae = dict()
		cfg_tr = dict(
			temp_anneal_portion=0.0,
			temp_stop=1.0,
		)
	elif model_type == 'categorical':
		cfg_vae = dict()
		cfg_tr = dict(
			temp_stop=0.1,
		)
	else:
		raise ValueError(model_type)

	# finalize main cfg
	cfg_vae = {**cfg_vae, **_archi2attr(archi_type)}
	# use n_features as: n_latenst or n_categories
	linear_decoder = cfg_vae['dec_type'] == 'lin'
	n_features = 512 if linear_decoder else 10
	n_channels = 16 if cfg_vae['enc_type'] == 'mlp' else 32
	cfg_vae = {
		'dataset': dataset,
		'n_ch': n_channels,
		'n_latents': n_features,
		**cfg_vae,
	}
	if model_type == 'categorical':
		cfg_vae['n_latents'] = 1
		cfg_vae['n_categories'] = n_features
	# init: dist & scale
	cfg_vae.update(dict(
		init_dist='Normal',
		init_scale=0.05 if
		linear_decoder
		else 0.1,
	))
	# init: prior_clamp
	if linear_decoder and 'prior_clamp' in cfg_vae:
		cfg_vae['prior_clamp'] = -4

	########################
	# trainer cfgs
	########################
	if dataset in ['vH16', 'CIFAR16']:
		cfg_tr = dict(
			**cfg_tr,
			lr=0.005,
			batch_size=1000,
			epochs=3000 if dataset == 'vH16' else 1500,
			grad_clip=None,
		)

	elif dataset.endswith('MNIST'):
		epochs = 400
		if dataset == 'EMNIST':
			epochs = 200
		cfg_tr = dict(
			**cfg_tr,
			lr=0.002,
			epochs=epochs,
			batch_size=100,
			warm_restart=0,
			grad_clip=1000,
		)

	elif dataset == 'BALLS':
		raise NotImplementedError(dataset)

	elif dataset == 'CIFAR10':
		raise NotImplementedError(dataset)

	else:
		raise ValueError(dataset)

	# finalize trainer cfg
	cfg_tr['kl_const_portion'] = 0.0 \
		if linear_decoder else 0.01

	return cfg_vae, cfg_tr


def _archi2attr(architecture: str):
	# get rid of the <bra|ket>
	if architecture.startswith('<'):
		architecture = architecture[1:]
	if architecture.endswith('>'):
		architecture = architecture[:-1]
	enc, dec = architecture.split('|')
	# enc
	if '+b' in enc:
		enc = enc.split('+')[0]
		enc_bias = True
	else:
		enc_bias = False
	# dec
	if '+b' in dec:
		dec = dec.split('+')[0]
		dec_bias = True
	else:
		dec_bias = False
	# attrs
	archi_attrs = dict(
		enc_type=str(enc),
		dec_type=str(dec),
		enc_bias=enc_bias,
		dec_bias=dec_bias,
	)
	return archi_attrs


CFG_CLASSES = {
	'poisson': ConfigPoisVAE,
	'gaussian': ConfigGausVAE,
	'laplace': ConfigLapVAE,
	'categorical': ConfigCatVAE,
}
