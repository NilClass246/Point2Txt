import torch
import torch.nn as nn
import time
from tools import builder
from utils.logger import *
from utils.config import *
from argparse import Namespace

args = Namespace(
    config="cfgs/ModelNet_models/PointTransformer.yaml",
    launcher="none",
    local_rank=0,
    num_workers=4,
    seed=0,
    deterministic=True,
    sync_bn=False,
    exp_name="test_random",
    start_ckpts=None,
    ckpts="./checkpoints/PointTransformer_ModelNet1024points.pth",
    val_freq=1,
    resume=False,
    test=True,
    finetune_model=False,
    scratch_model=False,
    label_smoothing=False,
    mode=None,
    way=-1,
    shot=-1,
    fold=-1,
    experiment_path="./experiments/PointTransformer/ModelNet_models/test_random",
    tfboard_path="./experiments/PointTransformer/ModelNet_models/TFBoard/test_random",
    log_name="PointTransformer",
    use_gpu=True,
    distributed = False
)

# WIP function to load Point-BERT model
def load_point_BERT():
    # prepare config
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    config = get_config(args, logger = logger)
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)


    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    base_model.eval()

    return base_model

if __name__ == "__main__":
    model = load_point_BERT()
    print(model)