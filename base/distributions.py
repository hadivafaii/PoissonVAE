from base.utils_model import *
dists.Distribution.set_default_validate_args(False)


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


# noinspection PyTypeChecker
class Poisson:
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 1.0,
			n_exp: int = 263,
			clamp: float = 5.3,
	):
		assert temp >= 0
		self.t = temp
		self.n = n_exp
		self.c = clamp
		self._init(log_rate)

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	def rsample(self, hard: bool = False):
		x = self.exp.rsample((self.n,))  # inter-event times
		times = torch.cumsum(x, dim=0)   # arrival times of events

		indicator = times < 1.0
		z_hard = indicator.sum(0).float()

		if self.t > 0:
			indicator = torch.sigmoid(
				(1.0 - times) / self.t)
			z = indicator.sum(0).float()
		else:
			z = z_hard

		if hard:
			return z + (z_hard - z).detach()
		return z

	def sample(self):
		return torch.poisson(self.rate).float()

	def log_p(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)

	def _init(self, log_rates):
		eps = torch.finfo(torch.float32).eps
		log_rates = softclamp_upper(log_rates, self.c)
		self.rate = torch.exp(log_rates) + eps
		self.exp = dists.Exponential(self.rate)
		return


def softclamp_sym(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


def softclamp_upper(x: torch.Tensor, c: float):
	return c - F.softplus(c - x)


def softclamp(x: torch.Tensor, upper: float, lower: float = 0.0):
	return lower + F.softplus(x - lower) - F.softplus(x - upper)
