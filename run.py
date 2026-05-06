import argparse
import os
import warnings
import torch

# print("Torch version:", torch.__version__)

import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from models.respace import SpacedDiffusion

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    # try:
    if opt['phase'] == 'train':
        model.train()
    else:
        model.test()
    # finally:
    #     # phase_writer.close()
    #     pass
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/NAS_data/yjy/palette_color_hint/config/colorization_sar2opt.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str,  default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='7')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('--color_hint_root', type=str, default=None, help='Default sparse color hint RGB directory (fallback for all phases).')
    parser.add_argument('--color_mask_root', type=str, default=None, help='Default binary hint mask directory (fallback for all phases).')
    parser.add_argument('--train_color_hint_root', type=str, default=None, help='Sparse color hint RGB directory for train dataset.')
    parser.add_argument('--train_color_mask_root', type=str, default=None, help='Binary hint mask directory for train dataset.')
    parser.add_argument('--val_color_hint_root', type=str, default=None, help='Sparse color hint RGB directory for val dataset.')
    parser.add_argument('--val_color_mask_root', type=str, default=None, help='Binary hint mask directory for val dataset.')
    parser.add_argument('--test_color_hint_root', type=str, default=None, help='Sparse color hint RGB directory for test dataset.')
    parser.add_argument('--test_color_mask_root', type=str, default=None, help='Binary hint mask directory for test dataset.')

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    def _set_hint_roots(phase_name, hint_root, mask_root):
        if phase_name not in opt['datasets']:
            return
        if hint_root is not None:
            opt['datasets'][phase_name]['which_dataset']['args']['color_hint_root'] = hint_root
        if mask_root is not None:
            opt['datasets'][phase_name]['which_dataset']['args']['color_mask_root'] = mask_root

    # phase-specific override first
    _set_hint_roots('train', args.train_color_hint_root, args.train_color_mask_root)
    _set_hint_roots('val', args.val_color_hint_root, args.val_color_mask_root)
    _set_hint_roots('test', args.test_color_hint_root, args.test_color_mask_root)

    # fallback default for any phase that did not get explicit override
    for _phase in ['train', 'val', 'test']:
        if _phase not in opt['datasets']:
            continue
        ds_args = opt['datasets'][_phase]['which_dataset']['args']
        if args.color_hint_root is not None and ds_args.get('color_hint_root') is None:
            ds_args['color_hint_root'] = args.color_hint_root
        if args.color_mask_root is not None and ds_args.get('color_mask_root') is None:
            ds_args['color_mask_root'] = args.color_mask_root
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)