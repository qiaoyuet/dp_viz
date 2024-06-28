import logging
import os.path
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.train_utils import eval_model
from pactl.nn import create_model
from pactl.optim.third_party.functional_warm_up import LinearWarmupScheduler
from pactl.optim.schedulers import construct_stable_cosine
from pactl.optim.schedulers import construct_warm_stable_cosine


def train(net, loader, criterion, optim, device=None, log_dir=None, epoch=None,
          non_private=True, dp_virtual_batch_size=128):
    net.train()

    if non_private:
        for i, (X, Y) in tqdm(enumerate(loader), leave=False):
            X, Y = X.to(device), Y.to(device)
            optim.zero_grad()
            f_hat = net(X)
            loss = criterion(f_hat, Y)
            loss.backward()
            optim.step()

            if log_dir is not None and i % 100 == 0:
                metrics = {'epoch': epoch, 'mini_loss': loss.detach().item()}
                logging.info(metrics, extra=dict(wandb=True, prefix='sgd/train'))
    else:
        with BatchMemoryManager(
                data_loader=loader,
                max_physical_batch_size=dp_virtual_batch_size,
                optimizer=optim
        ) as memory_safe_data_loader:
            for i, (X, Y) in tqdm(enumerate(memory_safe_data_loader), leave=False):
                X, Y = X.to(device), Y.to(device)
                optim.zero_grad()
                f_hat = net(X)
                loss = criterion(f_hat, Y)
                loss.backward()
                optim.step()

                if log_dir is not None and i % 100 == 0:
                    metrics = {'epoch': epoch, 'mini_loss': loss.detach().item()}
                    logging.info(metrics, extra=dict(wandb=True, prefix='sgd/train'))


def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
         dataset=None, train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='resnet18k', base_width=None,
         batch_size=128, optimizer='adam', lr=1e-3, momentum=.9, weight_decay=5e-4, epochs=0,
         intrinsic_dim=0, intrinsic_mode='filmrdkron',
         warmup_epochs=0, warmup_lr=.1, non_private=True, target_epsilon=-1, dp_C=1.0, dp_noise=-1,
         dp_virtual_batch_size=128, ckpt_every=[], exp_name='tmp'):
    random_seed_all(seed)

    train_data, test_data = get_dataset(
        dataset, root=data_dir,
        train_subset=train_subset,
        label_noise=label_noise,
        indices_path=indices_path)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=not distributed,
                              sampler=DistributedSampler(train_data) if distributed else None)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                             sampler=DistributedSampler(test_data) if distributed else None)

    if log_dir is not None:
        ckpt_path = os.path.join(log_dir, exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

    net = create_model(model_name=model_name, num_classes=train_data.num_classes, in_chans=train_data[0][0].size(0),
                       base_width=base_width,
                       seed=seed, intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode,
                       cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir, exp_name=exp_name)
    if distributed:
        # net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[device_id], broadcast_buffers=True)

    # removed if not non_private condition, use same model for priv and non-priv training
    errors = ModuleValidator.validate(net, strict=False)
    print(errors)
    net = ModuleValidator.fix(net)

    criterion = nn.CrossEntropyLoss()
    if optimizer == 'sgd':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    elif optimizer == 'ssc':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = construct_stable_cosine(
            optimizer=optimizer, lr_max=lr, lr_min=lr / 100., epochs=(100, epochs - 100))
    elif optimizer == 'wsc':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = construct_warm_stable_cosine(
            optimizer=optimizer, lrs=(lr / 100., lr, lr / 10.),
            epochs=(5, 75, epochs - 80))
    elif optimizer == 'awsc':
        optimizer = Adam(net.parameters(), lr=lr)
        optim_scheduler = construct_warm_stable_cosine(
            optimizer=optimizer, lrs=(lr / 100., lr, lr / 10.),
            epochs=(5, 75, epochs - 80))
    elif optimizer == 'adam':
        optimizer = Adam(net.parameters(), lr=lr)
        optim_scheduler = None
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif optimizer == 'adamw':
        optimizer = AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100.)
    elif optimizer == 'sgd_cos':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100.)
    elif optimizer == 'sgd_cos10':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10.)
    elif optimizer == 'sgd_only':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = None
    else:
        raise NotImplementedError

    if warmup_epochs > 0:
        optim_scheduler = LinearWarmupScheduler(optimizer,
                                                warm_epochs=[warmup_epochs], lr_goal=[warmup_lr],
                                                scheduler_after=[optim_scheduler])
    if not non_private:
        # Privacy engine
        privacy_engine = PrivacyEngine()
        if target_epsilon > 0 and dp_noise < 0:
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=net,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=1e-5,
                max_grad_norm=dp_C,
            )
            # print(f"Using sigma={optimizer.noise_multiplier} and C={dp_C}")
            logging.info({'dp_noise': optimizer.noise_multiplier, 'dp_C': dp_C},
                         extra=dict(wandb=True, prefix='sgd/train'))
        elif dp_noise > 0 and target_epsilon < 0:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=net,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=dp_noise,
                max_grad_norm=dp_C,
            )
            logging.info({'dp_noise': dp_noise, 'dp_C': dp_C},
                         extra=dict(wandb=True, prefix='sgd/train'))
        else:
            raise ValueError('Either specify noise multiplier or target epsilon.')

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        if distributed:
            train_loader.sampler.set_epoch(e)

        train(net, train_loader, criterion, optimizer, device=device_id,
              log_dir=log_dir, epoch=e, non_private=non_private, dp_virtual_batch_size=dp_virtual_batch_size)

        if optim_scheduler is not None:
            optim_scheduler.step()

        train_metrics = eval_model(net, train_loader, criterion, device_id=device_id, distributed=distributed)
        test_metrics = eval_model(net, test_loader, criterion, device_id=device_id, distributed=distributed)

        if log_dir is not None:
            logging.info(train_metrics, extra=dict(wandb=True, prefix='sgd/train'))
            logging.info(test_metrics, extra=dict(wandb=True, prefix='sgd/test'))

            if not non_private:
                logging.info({'dp_epsilon': privacy_engine.get_epsilon(delta=1e-5)},
                             extra=dict(wandb=True, prefix='sgd/train'))

            # ckpt_path = os.path.join(log_dir, exp_name)
            # if not os.path.exists(ckpt_path):
            #     os.mkdir(ckpt_path)

            # save intermediate ckpts
            if len(ckpt_every) > 0 and e in ckpt_every:
                torch.save(net.state_dict(), Path(log_dir) / exp_name / 'interm_model_e{}.pt'.format(e))

            if test_metrics['acc'] > best_acc_so_far:
                best_acc_so_far = test_metrics['acc']

                wandb.run.summary['sgd/test/best_epoch'] = e
                wandb.run.summary['sgd/test/best_acc'] = best_acc_so_far
                wandb.run.summary['sgd/train/best_acc'] = train_metrics['acc']
                logging.info(f"Epoch {e}: {train_metrics['acc']:.4f} (Train) / {best_acc_so_far:.4f} (Test)")

                torch.save(net.state_dict(), Path(log_dir) / exp_name / 'best_sgd_model.pt')

            torch.save(net.state_dict(), Path(log_dir) / exp_name / 'sgd_model.pt')
            wandb.save('*.pt')  ## NOTE: to upload immediately.


def entrypoint(log_dir=None, exp_group='tmp', exp_name='tmp', **kwargs):
    world_size, rank, device_id = maybe_launch_distributed()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(device_id)

    ## Only setup logging from one process (rank = 0).
    log_dir = set_logging(log_dir=log_dir, exp_group=exp_group, exp_name=exp_name) if rank == 0 else None
    if rank == 0:
        logging.info(f'Working with {world_size} process(es).')

    main(**kwargs, log_dir=log_dir, distributed=(world_size > 1), device_id=device_id, exp_name=exp_name)

    if rank == 0:
        finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)
