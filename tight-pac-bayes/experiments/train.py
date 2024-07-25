import logging
import os.path
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import time
import numpy as np

from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset, get_dataset_with_canaries
from pactl.train_utils import eval_model
from pactl.nn import create_model, create_model_tmp
from pactl.optim.third_party.functional_warm_up import LinearWarmupScheduler
from pactl.optim.schedulers import construct_stable_cosine
from pactl.optim.schedulers import construct_warm_stable_cosine

from experiments.auditing_utils import find_O1_pred, generate_auditing_data


def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
         dataset=None, train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, ckpt_name=None, transfer=False, model_name='resnet18k', base_width=None,
         batch_size=128, optimizer='adam', lr=1e-3, momentum=.9, weight_decay=5e-4, epochs=0,
         intrinsic_dim=0, intrinsic_mode='filmrdkron',
         warmup_epochs=0, warmup_lr=.1, non_private=True, target_epsilon=-1, dp_C=1.0, dp_noise=-1,
         dp_virtual_batch_size=128, ckpt_every=[], exp_name='tmp', eval_every=1000,
         audit=False, audit_size=1000):

    if log_dir is not None:
        ckpt_path = os.path.join(log_dir, exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

    random_seed_all(seed)

    # inserting canaries to both train and test data
    # train_data, test_data = get_dataset(
    #     dataset, root=data_dir,
    #     train_subset=train_subset,
    #     label_noise=label_noise,
    #     indices_path=indices_path)
    train_data, test_data = get_dataset_with_canaries(
        dataset, root=data_dir,
        train_subset=train_subset,
        label_noise=label_noise,
        indices_path=indices_path)

    if audit:
        mem_data, non_mem_data = generate_auditing_data(train_data, audit_size=audit_size)
        mem_loader = DataLoader(mem_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        non_mem_loader = DataLoader(non_mem_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        logging.info({'num_member': len(mem_data), 'num_non_member': len(non_mem_data)},
                     extra=dict(wandb=True, prefix='audit'))

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                              shuffle=not distributed,
                              sampler=DistributedSampler(train_data) if distributed else None)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                             sampler=DistributedSampler(test_data) if distributed else None)  # use test data as non-mem audit set

    # net = create_model(model_name=model_name, num_classes=train_data.num_classes, in_chans=train_data[0][0].size(0),
    #                    base_width=base_width,
    #                    seed=seed, intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode,
    #                    cfg_path=cfg_path, ckpt_name=ckpt_name,
    #                    transfer=transfer, device_id=device_id, log_dir=log_dir, exp_name=exp_name
    #                    )
    net = create_model_tmp(model_name=model_name, num_classes=train_data.num_classes, in_chans=train_data[0][0].size(0),
                       base_width=base_width,
                       seed=seed, intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode,
                       cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir)
    if distributed:
        # net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[device_id], broadcast_buffers=True)

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
        privacy_engine = PrivacyEngine(accountant='prv')
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
            logging.info({'dp_noise': optimizer.noise_multiplier, 'dp_C': dp_C},
                         extra=dict(wandb=True, prefix='train'))
        elif dp_noise > 0 and target_epsilon < 0:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=net,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=dp_noise,
                max_grad_norm=dp_C,
            )
            logging.info({'dp_noise': dp_noise, 'dp_C': dp_C},
                         extra=dict(wandb=True, prefix='train'))
        else:
            raise ValueError('Either specify noise multiplier or target epsilon.')

    if audit:
        # compute initial loss value as auditing score baseline (optional)
        # make sure shuffle is False in data loaders
        _, init_mem_losses = eval_model(net, mem_loader, criterion, device_id=device_id,
                                        distributed=distributed, audit=True)
        _, init_non_mem_losses = eval_model(net, non_mem_loader, criterion, device_id=device_id,
                                            distributed=distributed, audit=True)

    best_train_acc_so_far = 0.
    best_test_acc_so_far = 0.
    step_counter = 0
    for e in tqdm(range(epochs)):
        net.train()
        if distributed:
            train_loader.sampler.set_epoch(e)

        if non_private:
            for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
                X, Y = X.to(device_id), Y.to(device_id)
                optimizer.zero_grad()
                f_hat = net(X)
                loss = criterion(f_hat, Y)
                loss.backward()
                optimizer.step()
                step_counter += 1

                # if log_dir is not None and i % 100 == 0:
                #     metrics = {'mini_loss': loss.detach().item()}
                #     logging.info(metrics, extra=dict(wandb=True, prefix='train'))

                if log_dir is not None and step_counter % eval_every == 0:
                    # t0 = time.time()
                    train_metrics, _ = eval_model(net, train_loader, criterion, device_id=device_id,
                                                  distributed=distributed, audit=False)
                    test_metrics, _ = eval_model(net, test_loader, criterion, device_id=device_id,
                                                 distributed=distributed, audit=False)
                    # t1 = time.time()
                    # print("++++++++ logging eval +++++++++: {}".format(t1 - t0))
                    train_metrics.update({'epoch': e, 'step': step_counter})
                    logging.info(train_metrics, extra=dict(wandb=True, prefix='train'))
                    logging.info(test_metrics, extra=dict(wandb=True, prefix='test'))

                    if audit:
                        # # save interm ckpts
                        torch.save(net.state_dict(),
                                   Path(log_dir) / exp_name / 'interm_model_s{}.pt'.format(step_counter))

                        # t0 = time.time()
                        _, cur_mem_losses = eval_model(net, mem_loader, criterion, device_id=device_id,
                                                       distributed=distributed, audit=True)
                        _, cur_non_mem_losses = eval_model(net, non_mem_loader, criterion, device_id=device_id,
                                                           distributed=distributed, audit=True)
                        # t1 = time.time()
                        # print("++++++++ audit eval +++++++++: {}".format(t1 - t0))
                        mem_losses = np.array(cur_mem_losses) - np.array(init_mem_losses)
                        non_mem_losses = np.array(cur_non_mem_losses) - np.array(init_non_mem_losses)
                        audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                        logging.info(audit_metrics, extra=dict(wandb=True, prefix='audit'))
                        # t2 = time.time()
                        # print("++++++++ audit +++++++++: {}".format(t2 - t1))

        else:
            with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=dp_virtual_batch_size,
                    optimizer=optimizer
            ) as memory_safe_data_loader:
                for i, (X, Y) in tqdm(enumerate(memory_safe_data_loader), leave=False):
                    X, Y = X.to(device_id), Y.to(device_id)
                    optimizer.zero_grad()
                    f_hat = net(X)
                    loss = criterion(f_hat, Y)
                    loss.backward()
                    optimizer.step()
                    step_counter += 1

                    # if log_dir is not None and i % 100 == 0:
                    #     metrics = {'mini_loss': loss.detach().item()}
                    #     logging.info(metrics, extra=dict(wandb=True, prefix='train'))

                    if log_dir is not None and step_counter % eval_every == 0:
                        train_metrics, _ = eval_model(net, train_loader, criterion, device_id=device_id,
                                                      distributed=distributed, audit=False)
                        test_metrics, _ = eval_model(net, test_loader, criterion, device_id=device_id,
                                                     distributed=distributed, audit=False)
                        dp_epsilon = privacy_engine.get_epsilon(delta=1e-5)
                        train_metrics.update({'epoch': e, 'step': step_counter, 'dp_epsilon': dp_epsilon})
                        logging.info(train_metrics, extra=dict(wandb=True, prefix='train'))
                        logging.info(test_metrics, extra=dict(wandb=True, prefix='test'))

                        if audit:
                            # t0 = time.time()
                            _, mem_losses = eval_model(net, mem_loader, criterion, device_id=device_id,
                                                       distributed=distributed, audit=True)
                            _, non_mem_losses = eval_model(net, non_mem_loader, criterion, device_id=device_id,
                                                           distributed=distributed, audit=True)
                            audit_metrics = find_O1_pred(mem_losses, non_mem_losses)
                            logging.info(audit_metrics, extra=dict(wandb=True, prefix='audit'))
                            # t1 = time.time()
                            # print("+++++++++++++++++: {}".format(t1 - t0))

        if optim_scheduler is not None:
            optim_scheduler.step()

        # epoch wise logging
        train_metrics, _ = eval_model(net, train_loader, criterion, device_id=device_id, distributed=distributed)
        # test_metrics, _ = eval_model(net, test_loader, criterion, device_id=device_id, distributed=distributed)
        logging.info(train_metrics, extra=dict(wandb=True, prefix='per_epoch_metrics'))
        # logging.info(test_metrics, extra=dict(wandb=True, prefix='per_epoch_metrics'))
        # if test_metrics['acc'] > best_test_acc_so_far:
        #     best_acc_so_far = test_metrics['acc']
        #     logging.info({'best_test_epoch': e, 'best_test_acc': best_acc_so_far},
        #                  extra=dict(wandb=True, prefix='test'))
        #     # torch.save(net.state_dict(), Path(log_dir) / exp_name / 'best_sgd_model.pt')
        # if train_metrics['acc'] > best_train_acc_so_far:
        #     best_acc_so_far = train_metrics['acc']
        #     logging.info({'best_train_epoch': e, 'best_train_acc': best_acc_so_far},
        #                  extra=dict(wandb=True, prefix='train'))
            # torch.save(net.state_dict(), Path(log_dir) / exp_name / 'sgd_model.pt')
        # wandb.save('*.pt')  ## NOTE: to upload immediately.


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
