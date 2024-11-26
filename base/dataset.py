from .utils_model import *
from torch.utils.data import Dataset
import torchvision


class ToDevice:
	def __init__(self, device):
		self.device = device

	def __call__(self, x):
		return x.to(self.device)


def make_dataset(
		dataset: str,
		load_dir: str,
		device: torch.device,
		**kwargs, ):

	tst = None

	if dataset == 'vH16':
		defaults = dict(n_blocks=100, vld_portion=0.2)
		kwargs = setup_kwargs(defaults, kwargs)

		data = pjoin(
			load_dir,
			'DOVES',
			dataset,
			'processed.npy',
		)
		data = np.load(data)

		trn_inds, vld_inds = split_data(
			n_samples=len(data),
			n_blocks=kwargs['n_blocks'],
			vld_portion=kwargs['vld_portion'],
		)
		trn, vld = data[trn_inds], data[vld_inds]
		trn, vld = map(
			lambda a: torch.tensor(
				data=a,
				device=device,
				dtype=torch.float),
			[trn, vld],
		)
		trn = torch.utils.data.TensorDataset(trn)
		vld = torch.utils.data.TensorDataset(vld)

	elif dataset == 'CIFAR16':
		path = pjoin(load_dir, 'CIFAR10', 'xtract16')
		trn = np.load(pjoin(path, 'trn', 'processed.npy'))
		vld = np.load(pjoin(path, 'vld', 'processed.npy'))

		trn, vld = map(_to_device_fun(device), [trn, vld])
		trn = torch.utils.data.TensorDataset(trn)
		vld = torch.utils.data.TensorDataset(vld)

	elif dataset == 'MNIST':
		path = pjoin(load_dir, 'MNIST', 'processed')
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld = _load(path, device)
		else:
			# get transforms
			transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				ToDevice(device=device),
			])
			# make dataset
			kws = dict(root=load_dir, transform=transform)
			trn = torchvision.datasets.MNIST(train=True, **kws)
			vld = torchvision.datasets.MNIST(train=False, **kws)
			# process and save
			trn, vld = _process_and_save(trn, vld, path)

	elif dataset == 'CIFAR10':
		path = pjoin(load_dir, 'CIFAR10', 'processed')
		if os.path.isfile(pjoin(path, 'x_trn.npy')):
			trn, vld = _load(path, device)
		else:
			# get transforms
			defaults = dict(grey=False)
			kwargs = setup_kwargs(defaults, kwargs)
			transform = [
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Grayscale()
				if kwargs['grey'] else None,
				torchvision.transforms.Normalize(
					mean=(0.5,) if kwargs['grey'] else (0.5,) * 3,
					std=(0.5,) if kwargs['grey'] else (0.5,) * 3),
				ToDevice(device=device),
			]
			transform = torchvision.transforms.Compose([
				t for t in transform if t is not None
			])
			# make dataset
			kws = dict(
				root=pjoin(load_dir, 'CIFAR10'),
				transform=transform,
			)
			trn = torchvision.datasets.CIFAR10(train=True, **kws)
			vld = torchvision.datasets.CIFAR10(train=False, **kws)
			# process and save
			trn, vld = _process_and_save(trn, vld, path)

	elif dataset == 'BALLS':
		defaults = dict(npix=16, vld_split=0.25)
		kwargs = setup_kwargs(defaults, kwargs)

		path = pjoin(
			load_dir,
			'BALLS',
			f"npix-{kwargs['npix']}",
		)
		# load
		x_tst = np.load(pjoin(path, 'x_tst.npy'))
		z_tst = np.load(pjoin(path, 'z_tst.npy'))
		x = np.load(pjoin(path, 'x.npy'))
		z = np.load(pjoin(path, 'z.npy'))
		# split into trn / vld
		frac = 1 - kwargs['vld_split']
		idx = int(len(x) * frac)
		x_trn, x_vld = x[:idx], x[idx:]
		z_trn, z_vld = z[:idx], z[idx:]

		# to tensor
		x_trn, x_vld, x_tst, z_trn, z_vld, z_tst = map(
			_to_device_fun(device),
			[x_trn, x_vld, x_tst, z_trn, z_vld, z_tst],
		)
		# to dataset
		trn = torch.utils.data.TensorDataset(x_trn, z_trn)
		vld = torch.utils.data.TensorDataset(x_vld, z_vld)
		tst = torch.utils.data.TensorDataset(x_tst, z_tst)

	else:
		raise ValueError(dataset)

	return trn, vld, tst


def split_data(
		n_samples: int,
		n_blocks: int = 1,
		vld_portion: float = 0.2, ):
	assert 0 < vld_portion < 1
	indices = range(n_samples)
	block_size = len(indices) // n_blocks

	trn_inds, vld_inds = [], []
	for b in range(n_blocks):
		start = b * block_size
		if b == n_blocks - 1:
			end = len(indices)
		else:
			end = start + block_size

		block_inds = indices[start:end]
		vld_size = np.round(len(block_inds) * vld_portion)
		trn_size = len(block_inds) - int(vld_size)

		trn_inds.extend(block_inds[:trn_size])
		vld_inds.extend(block_inds[trn_size:])

	assert not set(trn_inds).intersection(
		vld_inds), "must be non-overlapping"
	trn_inds, vld_inds = map(
		lambda x: np.array(x),
		[trn_inds, vld_inds],
	)
	return trn_inds, vld_inds


def _load(load_dir, device):
	# load
	x_trn = np.load(pjoin(load_dir, 'x_trn.npy'))
	x_vld = np.load(pjoin(load_dir, 'x_vld.npy'))
	y_trn = np.load(pjoin(load_dir, 'y_trn.npy'))
	y_vld = np.load(pjoin(load_dir, 'y_vld.npy'))
	# to tensor
	x_trn, x_vld, y_trn, y_vld = map(
		_to_device_fun(device),
		[x_trn, x_vld, y_trn, y_vld],
	)
	trn = torch.utils.data.TensorDataset(x_trn, y_trn)
	vld = torch.utils.data.TensorDataset(x_vld, y_vld)
	return trn, vld


def _process_and_save(trn, vld, save_dir):
	# process
	x_trn, y_trn = _process(trn)
	x_vld, y_vld = _process(vld)
	# save
	os.makedirs(save_dir, exist_ok=True)
	_save = {
		'x_trn': tonp(x_trn),
		'x_vld': tonp(x_vld),
		'y_trn': tonp(y_trn),
		'y_vld': tonp(y_vld),
	}
	for name, obj in _save.items():
		save_obj(
			obj=obj,
			file_name=name,
			save_dir=save_dir,
			verbose=True,
			mode='npy',
		)
	# to dataset
	trn = torch.utils.data.TensorDataset(x_trn, y_trn)
	vld = torch.utils.data.TensorDataset(x_vld, y_vld)
	return trn, vld


def _process(dataset):
	loader = torch.utils.data.DataLoader(
		dataset, batch_size=len(dataset))
	return next(iter(loader))


def _to_device_fun(
		device: torch.device,
		dtype=torch.float, ):
	def _fun(a):
		return torch.tensor(
			data=a,
			dtype=dtype,
			device=device,
		)
	return _fun
