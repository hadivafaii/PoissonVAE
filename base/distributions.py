from base.utils_model import *
dists.Distribution.set_default_validate_args(False)


class Poisson:
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 0.0,
			clamp: float | None = None,
			indicator_approx: str = 'sigmoid',
			n_exp: int | str = 'infer',
			n_exp_p: float = 1e-3,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		assert indicator_approx in _INDICATOR_FNS
		self.indicator_approx = indicator_approx
		self.temp = temp
		self.clamp = clamp
		# setup rate & exp dist
		if self.clamp is not None:
			log_rate = softclamp_upper(
				log_rate, self.clamp)
		eps = torch.finfo(torch.float32).eps
		self.rate = torch.exp(log_rate) + eps
		self._exp = dists.Exponential(self.rate)
		# compute n_exp
		if n_exp == 'infer':
			n_exp = self._infer_n_exp(n_exp_p)
		self.n_exp = int(n_exp)

	def __repr__(self):
		parts = [
			f"rate: {self.rate.shape}",
			f"temp: {self.temp}",
			f"n_exp: {self.n_exp}",
		]
		return f"Poisson({', '.join(parts)})"

	@torch.no_grad()
	def _infer_n_exp(self, n_exp_p):
		max_rate = self.rate.max().item()
		n_exp = compute_n_exp(max_rate, n_exp_p)
		return int(n_exp)

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	# noinspection PyTypeChecker
	def rsample(self):
		if self.temp == 0.0:
			return self.sample()

		# (1) inter-event times
		x = self._exp.rsample((self.n_exp,))

		# (2) arrival t of events
		times = torch.cumsum(x, dim=0)

		# (3) compute raw logits
		# (input to the sigmoid-like function)
		# This maps the threshold time t=1 to 0
		# t = 1 - temp → logits = 1
		# t = 1 + temp → logits = -1
		logits = (1 - times) / self.temp

		# (4) events within [0, 1]
		fn = _INDICATOR_FNS.get(
			self.indicator_approx)
		indicator = fn(logits)

		# (5) soft event counts
		z = indicator.sum(0).to(
			dtype=self.rate.dtype)

		return z

	@torch.no_grad()
	def sample(self):
		return torch.poisson(self.rate).float()

	def log_prob(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)


# noinspection PyAbstractClass
class Categorical(dists.RelaxedOneHotCategorical):
	def __init__(
			self,
			logits: torch.Tensor,
			temp: float = 1.0,
			**kwargs,
	):
		self._categorical = None
		temp = max(temp, torch.finfo(torch.float).eps)
		super(Categorical, self).__init__(
			logits=logits, temperature=temp, **kwargs)

	@property
	def t(self):
		return self.temperature

	@property
	def mean(self):
		return self.probs

	@property
	def variance(self):
		return self.probs * (1 - self.probs)

	def kl(self, p: dists.Categorical = None):
		if p is None:
			probs = torch.full(
				size=self.probs.size(),
				fill_value=1 / self.probs.size(-1),
			)
			p = dists.Categorical(probs=probs)
		q = dists.Categorical(probs=self.probs)
		return dists.kl.kl_divergence(q, p)


# noinspection PyAbstractClass
class Laplace(dists.Laplace):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(log_scale, clamp)
		super(Laplace, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.t = temp
		self.c = clamp

	def kl(self, p: dists.Laplace = None):
		if p is not None:
			mean, scale = p.mean, p.scale
		else:
			mean, scale = 0, 1

		delta_m = torch.abs(self.mean - mean)
		delta_b = self.scale / scale
		term1 = delta_m / self.scale
		term2 = delta_m / scale

		kl = (
			delta_b * torch.exp(-term1) +
			term2 - torch.log(delta_b) - 1
		)
		return kl


# noinspection PyAbstractClass
class Normal(dists.Normal):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			seed: int = None,
			device: torch.device = None,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(log_scale, clamp)
		super(Normal, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.t = temp
		self.c = clamp
		self._init_rng(seed, device)

	def kl(self, p: dists.Normal = None):
		if p is None:
			term1 = self.mean
			term2 = self.scale
		else:
			term1 = (self.mean - p.mean) / p.scale
			term2 = self.scale / p.scale
		kl = 0.5 * (
			term1.pow(2) + term2.pow(2) +
			torch.log(term2).mul(-2) - 1
		)
		return kl

	@torch.inference_mode()
	def sample(self, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		samples = torch.normal(
			mean=self.loc.expand(shape),
			std=self.scale.expand(shape),
			generator=self.rng,
		)
		return samples

	def _init_rng(self, seed, device):
		if seed is not None:
			self.rng = torch.Generator(device)
			self.rng.manual_seed(seed)
		else:
			self.rng = None
		return


def compute_n_exp(rate: float, p: float = 1e-6):
	assert rate > 0.0, f"must be positive, got: {rate}"
	pois = sp_stats.poisson(rate)
	n_exp = pois.ppf(1.0 - p)
	return int(n_exp)


def softclamp(x: torch.Tensor, upper: float, lower: float = 0.0):
	return lower + F.softplus(x - lower) - F.softplus(x - upper)


def softclamp_sym(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


def softclamp_upper(x: torch.Tensor, c: float):
	return c - F.softplus(c - x)


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
	"""
	Piecewise linear approximation (Hard Sigmoid).
	Maps [-1, 1] linearly to [0, 1].
	Exact 0 for x < -1, Exact 1 for x > 1.
	"""
	return torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)


def cubic_smoothstep(x: torch.Tensor) -> torch.Tensor:
	"""
	Cubic Hermite interpolation (Smoothstep).
	Maps [-1, 1] to [0, 1] with C1 smoothness
	(zero derivative at boundaries).
	f(u) = 3u^2 - 2u^3, where u = (x+1)/2
	"""
	# 1. Normalize x from [-1, 1] to u in [0, 1]
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
	# 2. Cubic polynomial
	return 3 * u.pow(2) - 2 * u.pow(3)


def quintic_smoothstep(x: torch.Tensor) -> torch.Tensor:
	"""
	Quintic Hermite interpolation (C2 Smoothstep).
	Maps [-1, 1] to [0, 1] with C2 smoothness
	(zero 1st and 2nd derivatives at boundaries).
	f(u) = 6u^5 - 15u^4 + 10u^3, where u = (x+1)/2
	"""
	# 1. Normalize x from [-1, 1] to u in [0, 1]
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)

	# 2. Quintic polynomial: 6u^5 - 15u^4 + 10u^3
	# Optimized for evaluation: u^3 * (u * (6u - 15) + 10)
	return 6 * u.pow(5) - 15 * u.pow(4) + 10 * u.pow(3)


def cosine_sigmoid(x: torch.Tensor) -> torch.Tensor:
	"""
	Cosine-based smooth approximation.
	Maps [-1, 1] to [0, 1] with C_infinity smoothness inside the window.
	f(u) = 0.5 * (1 - cos(pi * u)), where u = (x+1)/2
	"""
	# 1. Normalize x from [-1, 1] to u in [0, 1]
	u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
	# 2. Cosine ease-in-out
	return 0.5 * (1.0 - torch.cos(torch.pi * u))


_INDICATOR_FNS = {
	'sigmoid': torch.sigmoid,
	'cosine': cosine_sigmoid,
	'linear': hard_sigmoid,
	'cubic': cubic_smoothstep,
	'quintic': quintic_smoothstep,
}
