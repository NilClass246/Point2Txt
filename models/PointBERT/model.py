import torch
import torch.nn as nn
import time
from argparse import Namespace
from .tools.builder import model_builder, load_model
from .utils.logger import *
from .utils.config import *

root = "./models/PointBERT"

args = Namespace(
    config=f"{root}/cfgs/ModelNet_models/PointTransformer.yaml",
    launcher="none",
    local_rank=0,
    num_workers=4,
    seed=0,
    deterministic=True,
    sync_bn=False,
    exp_name="test_random",
    start_ckpts=None,
    ckpts=f"{root}/checkpoints/PointTransformer_ModelNet1024points.pth",
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
    experiment_path=f"{root}/experiments/PointTransformer/ModelNet_models/test_random",
    tfboard_path=f"{root}/experiments/PointTransformer/ModelNet_models/TFBoard/test_random",
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

    # _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = model_builder(config.model)
    load_model(base_model, args.ckpts, logger = logger)
    base_model.eval()

    base_model.cls_head_finetune = torch.nn.Identity()
    for p in base_model.parameters():
        p.requires_grad = False

    return base_model