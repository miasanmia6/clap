import torch
import trainers
import multiprocessing
# 假设在这里引入了必要的模块
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
sys.path.insert(0, "/root/autodl-tmp/BayesLinearAdapter/CLAP-LaplaceGGN/Dasslpytorch")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.backends.cuda.matmul.allow_tf32 = False  # 防止隐式图构建



def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    for key, value in args.items():
        print(f"{key}: {value}")
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args['root']:
        cfg.DATASET.ROOT = args['root']

    if args['output_dir']:
        cfg.OUTPUT_DIR = args['output_dir']

    if args['resume']:
        cfg.RESUME = args['resume']

    if args['seed']:
        cfg.SEED = args['seed']

    if args['source_domains']:
        cfg.DATASET.SOURCE_DOMAINS = args['source_domains']

    if args['target_domains']:
        cfg.DATASET.TARGET_DOMAINS = args['target_domains']

    if args['transforms']:
        cfg.INPUT.TRANSFORMS = args['transforms']

    if args['trainer']:
        cfg.TRAINER.NAME = args['trainer']

    if args['backbone']:
        cfg.MODEL.BACKBONE.NAME = args['backbone']

    if args['head']:
        cfg.MODEL.HEAD.NAME = args['head']

    if args['num_shots']:
        cfg.SHOT = args['num_shots']
        
    if args['dataset']:
        cfg.DATASET_NAME = args['dataset']


def extend_cfg(cfg, flag, num_shots=1):
    from yacs.config import CfgNode as CN
    if flag == 'ADAPTER':
        cfg.TRAINER.ADAPTER = CN()
        cfg.TRAINER.ADAPTER.INIT = "ZS"
        cfg.TRAINER.ADAPTER.CONSTRAINT = "l2"
        cfg.TRAINER.ADAPTER.ENHANCED_BASE = "none"
        cfg.TRAINER.ADAPTER.PREC = "fp16"

        cfg.DATASET.SUBSAMPLE_CLASSES = "all"
        cfg.DATASET.NUM_SHOTS = num_shots
    if flag == 'Bayes':
        cfg.TRAINER.BayesADAPTER = CN()
        cfg.TRAINER.BayesADAPTER.INIT = "GAUSSIAN_PER_CLASS"
        cfg.TRAINER.BayesADAPTER.CONSTRAINT = "kl"
        cfg.TRAINER.BayesADAPTER.ENHANCED_BASE = "none"
        cfg.TRAINER.BayesADAPTER.PREC = "fp16"

        cfg.DATASET.SUBSAMPLE_CLASSES = "all"
        cfg.DATASET.NUM_SHOTS = num_shots

    if flag == 'LaplaceBayesADAPTER':
        cfg.TRAINER.LaplaceBayesADAPTER = CN()
        cfg.TRAINER.LaplaceBayesADAPTER.INIT = "ZS"
        cfg.TRAINER.LaplaceBayesADAPTER.CONSTRAINT = "l2"
        cfg.TRAINER.LaplaceBayesADAPTER.ENHANCED_BASE = "none"
        cfg.TRAINER.LaplaceBayesADAPTER.PREC = "fp16"

        cfg.DATASET.SUBSAMPLE_CLASSES = "all"
        cfg.DATASET.NUM_SHOTS = num_shots


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, flag=args['trainer'], num_shots=args['num_shots'])  # 修改这里

    if args['dataset_config_file']:
        cfg.merge_from_file(args['dataset_config_file'])

    if args['config_file']:
        cfg.merge_from_file(args['config_file'])

    reset_cfg(cfg, args)

    cfg.merge_from_list(args['opts'])
    if cfg.TRAINER.NAME=='ADAPTER':
        if "TipA" in cfg.TRAINER.ADAPTER.INIT:
            cfg.DATALOADER.TRAIN_X.SAMPLER = "SequentialSampler"
        else:
            cfg.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
    if cfg.TRAINER.NAME=='BayesADAPTER':
        if "TipA" in cfg.TRAINER.BayesADAPTER.INIT:
            cfg.DATALOADER.TRAIN_X.SAMPLER = "SequentialSampler"
        else:
            cfg.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
    if cfg.TRAINER.NAME=='LaplaceBayesADAPTER':
        if "TipA" in cfg.TRAINER.LaplaceBayesADAPTER.INIT:
            cfg.DATALOADER.TRAIN_X.SAMPLER = "SequentialSampler"
        else:
            cfg.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)


    # ✅ 正常评估流程（非贝叶斯）
    if args['eval_only']:
        trainer.load_model(args['model_dir'], cfg, epoch=args['load_epoch'])
        trainer.test()
        return

    # ✅ 正常训练流程
    if not args['no_train']:
        trainer.train()


if __name__ == "__main__":
    for shot in [1, 2, 4, 8, 16, 32]:
        for seed in [1, 2, 3]:
            dataset = "oxford_flowers"
            args = {
                'dataset': dataset,
                'root': "/root/autodl-tmp/BayesLinearAdapter/CLAP-LaplaceGGN/data",
                'num_shots': shot,
                'output_dir': f"BayesLinearAdapter/FINAL/debug/{dataset}/SGD_lr1e-1_B256_ep300/kl/{shot}shots/{seed}seed",
                'resume': "",
                'seed': seed,

                'source_domains': ['source_domain1', 'source_domain2'],
                'target_domains': ['target_domain1', 'target_domain2'],
                'transforms': ["random_resized_crop", "random_flip", "normalize"],
                'config_file': "configs/trainers/SGD_lr1e-1_B256_ep300.yaml",
                'dataset_config_file': f"configs/datasets/{dataset}.yaml",
                'trainer': "LaplaceBayesADAPTER",
                'backbone': "RN50",
                'head': "",
                'eval_only': False,
                'model_dir': "",
                'load_epoch': None,
                'no_train': False,
                'enhanced_base': "none",

                'DATALOADER': {
                    'TRAIN_X': {'BATCH_SIZE': 256},
                    'TEST': {'BATCH_SIZE': 500},
                    'NUM_WORKERS': 8
                },
                'INPUT': {
                    'SIZE': (224, 224),
                    'INTERPOLATION': "bicubic",
                    'PIXEL_MEAN': [0.48145466, 0.4578275, 0.40821073],
                    'PIXEL_STD': [0.26862954, 0.26130258, 0.27577711],
                    'TRANSFORMS': ["random_resized_crop", "random_flip", "normalize"]
                },
                'opts': [
                    'OPTIM.LR', 0.1,
                    'OPTIM.MAX_EPOCH', 300,
                    'OPTIM.MOMENTUM', 0.9,
                    'OPTIM.WEIGHT_DECAY', 0.0,
                    'OPTIM.LR_SCHEDULER', 'cosine',
                    'OPTIM.WARMUP_EPOCH', 100,
                    'OPTIM.WARMUP_TYPE', 'linear',
                    'OPTIM.WARMUP_CONS_LR', 1e-5,
                    'OPTIM.WARMUP_RECOUNT', True,
                    'DATALOADER.TRAIN_X.BATCH_SIZE', 256,
                    'DATALOADER.TEST.BATCH_SIZE', 500,
                    'DATALOADER.NUM_WORKERS', 8
                ],
                'TRAIN': {'PRINT_FREQ': 5}
            }

            main(args)

