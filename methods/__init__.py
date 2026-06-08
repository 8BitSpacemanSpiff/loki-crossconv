from .common.saver import TensorSaver as TensorSaver
from .common.logger import WandbLogger as WandbLogger
from .common.logger import NoOpLogger as NoOpLogger

G_TENSOR_SAVER = None
LOGGER = None
G_TIMERS = None
DIAGNOSTICS = {}

def init_tensor_saver(tensor_dir):
    global G_TENSOR_SAVER 
    G_TENSOR_SAVER = TensorSaver(tensor_dir)

def init_logger(args):
    global LOGGER
    if args.use_wandb:
        LOGGER = WandbLogger(args)
    else:
        LOGGER = NoOpLogger(args)

def finish_logger():
    global LOGGER
    if LOGGER is not None:
        LOGGER.finish()

def record_diagnostic(name, layer_idx, value):
    DIAGNOSTICS.setdefault(name, {}).setdefault(layer_idx, []).append(float(value))

def print_diagnostics_summary():
    if not DIAGNOSTICS:
        return
    print("=== Diagnostics summary ===")
    for name in sorted(DIAGNOSTICS):
        layer_means = []
        count = 0
        for values in DIAGNOSTICS[name].values():
            if values:
                layer_means.append(sum(values) / len(values))
                count += len(values)
        if layer_means:
            overall = sum(layer_means) / len(layer_means)
            print(f"{name}: overall={overall:.6f} layers={len(layer_means)} samples={count}")

