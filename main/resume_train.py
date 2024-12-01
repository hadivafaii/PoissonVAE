from utils.generic import *
from .train_vae import load_model, save_fit_info


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"device",
		choices=range(torch.cuda.device_count()),
		help='cuda:device',
		type=int,
	)
	parser.add_argument(
		"model_name",
		help='model string?',
		type=str,
	)
	parser.add_argument(
		"fit_name",
		help='fit string?',
		type=str,
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
	return parser.parse_args()


def _main():
	args = _setup_args()
	device = f"cuda:{args.device}"

	tr, meta = load_model(
		model_name=args.model_name,
		fit_name=args.fit_name,
		device=device,
		strict=True,
	)
	epochs = range(  # remaining epochs
		meta['checkpoint'],
		tr.cfg.epochs,
	)
	if not len(epochs):
		return

	args['archi'] = tr.model.cfg.attr2archi()

	if args.cudnn_bench:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.benchmark_limit = 0

	if args.verbose:
		print(args)
		tr.model.print()
		msg = tr.model.cfg.name() + \
			f"\n{tr.cfg.name()}" + \
			f"_({tr.model.timestamp})\n"
		print(msg)

	start = now(True)
	if not args.dry_run:
		tr.train(epochs=epochs, fresh_fit=False)
		save_fit_info(vars(args), tr, start)
	return


if __name__ == "__main__":
	_main()
