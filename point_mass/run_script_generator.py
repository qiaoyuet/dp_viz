import argparse
import itertools
import os


def main_sim(args):
    # sim
    # command_template = "nohup python -u sim.py --N {nsamples} --lr {lr} --n_epoch {nepochs} --l2_reg 0 --no_plot " \
    #                    "--target_epsilon 3.0 --dp_C {dpc} " \
    #                    "--exp_group 1129 " \
    #                    "--exp_name {name} &> tmp.out&"
    # mnist
    # command_template = "nohup python -u mnist.py --lr {lr} --n_epoch 50000 --batch_size 2048 " \
    #                    "--eval_every 10 --audit --audit_proportion 0.2 --non_priv " \
    #                    "--exp_group sim_mnist_1200_1230 --exp_name {name} &> tmp.out&"
    command_template = "nohup python -u mnist.py --lr {lr} --n_epoch 50000 --batch_size 2048 " \
                       "--eval_every 10 --audit --audit_proportion 0.2 " \
                       "--dp_C 1.0 --dp_noise {dpn} " \
                       "--exp_group sim_mnist_priv_1200_1230 --exp_name {name} &> tmp2.out&"

    hyperparam_dict = {
        # 'nsamples': [int(item) for item in args.nsamples.split(',')],
        # 'nepochs': [int(item) for item in args.nepochs.split(',')],
        'lr': [float(item) for item in args.lr.split(',')],
        # 'ap': [float(item) for item in args.ap.split(',')],
        # 'dpc': [float(item) for item in args.dpc.split(',')],
        'dpn': [float(item) for item in args.dpn.split(',')],
    }
    keys, values = zip(*hyperparam_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for tmp_dict in permutations_dicts:
        # name = 'eps2_n{}_e{}_lr{}_c{}'.format(
        #     tmp_dict['nsamples'], tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['dpc']
        # )
        # name = 'nonpriv_e{}_lr{}_bs2048_ap{}'.format(
        #     # tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['ap']
        #     50000, tmp_dict['lr'], 0.2
        # )
        # name = 'priv3_e{}_lr{}_c{}'.format(
        #     tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['dpc']
        # )
        name = 'priv_e{}_lr{}_c{}_n{}'.format(
            50000, tmp_dict['lr'], 1, tmp_dict['dpn']
        )
        python_command = command_template.format(
            # nsamples=tmp_dict['nsamples'],
            # nepochs=tmp_dict['nepochs'],
            lr=tmp_dict['lr'],
            # ap=tmp_dict['ap'],
            # dpc=tmp_dict['dpc'],
            dpn=tmp_dict['dpn'],
            name=name
        )

        print(python_command)

        # script_content = dedent(run_script_template.format(python_command=python_command))
        # script_name = "{}.sh".format(name)
        # with open(os.path.join(args.logs, script_name), "w") as f:
        #     f.write(script_content)
        # print("sbatch {}".format(script_name))


def main_distill(args):
    command_template = "nohup python -u distill.py " \
                       "--load_exp_name nonpriv_n100_lr0.01_e3000 --load_step {load_step} " \
                       "--n_epoch_stu {nepochs} --lr {lr} --alpha {alpha} " \
                       "--exp_group 1203_student --exp_name {name} --no_plot --non_priv &> tmp.out&"
    # command_template = "nohup python -u distill.py " \
    #                    "--load_exp_name eps2_n100_e3000_lr0.01_c0.1 --load_step {load_step} " \
    #                    "--n_epoch_stu {nepochs} --lr {lr} --alpha {alpha} " \
    #                    "--exp_group 1203_student --exp_name {name} --no_plot &> tmp.out&"

    hyperparam_dict = {
        'load_step': [int(item) for item in args.load_step.split(',')],
        'nepochs': [int(item) for item in args.nepochs.split(',')],
        'lr': [float(item) for item in args.lr.split(',')],
        'alpha': [float(item) for item in args.alpha.split(',')],
    }
    keys, values = zip(*hyperparam_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tmp_count = 0
    for tmp_dict in permutations_dicts:
        name = 'stu_non_priv_e{}_lr{}_a{}_s{}'.format(
            tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['alpha'], tmp_dict['load_step']
        )
        # name = 'stu_eps2_e{}_lr{}_a{}_s{}'.format(
        #     tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['alpha'], tmp_dict['load_step']
        # )
        python_command = command_template.format(
            nepochs=tmp_dict['nepochs'],
            lr=tmp_dict['lr'], alpha=tmp_dict['alpha'], name=name,
            load_step=tmp_dict['load_step']
        )
        print(python_command)
        tmp_count += 1

        if tmp_count > 15:
            print("\n")
            tmp_count = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", default="100")
    parser.add_argument("--nepochs", default="10,100,500,1000,5000,10000,50000")
    parser.add_argument("--lr", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--ap", default="0.3,0.4,0.5,0.6")
    parser.add_argument("--dpc", default="1.0")
    parser.add_argument("--dpn", default="0.5,1.0,5.0,10.0,50.0")
    parser.add_argument("--alpha", default="0.1,0.3,0.5,0.7,0.9")
    # parser.add_argument("--load_step", default="100,1500,2950")
    parser.add_argument("--load_step", default="10,150,2990")
    args = parser.parse_args()
    main_sim(args)
    # main_distill(args)
