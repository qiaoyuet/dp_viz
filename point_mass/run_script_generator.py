import argparse
import itertools


def main(args):
    command_template = "nohup python -u sim.py --N {nsamples} --lr {lr} --n_epoch {nepochs} --l2_reg 0 --no_plot " \
                       "--target_epsilon 2.0 --dp_C {dpc} " \
                       "--exp_group 1129 " \
                       "--exp_name {name} &> tmp.out&"

    hyperparam_dict = {
        'nsamples': [int(item) for item in args.nsamples.split(',')],
        'nepochs': [int(item) for item in args.nepochs.split(',')],
        'lr': [float(item) for item in args.lr.split(',')],
        'dpc': [float(item) for item in args.dpc.split(',')],
    }
    keys, values = zip(*hyperparam_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for tmp_dict in permutations_dicts:
        name = 'eps2_n{}_e{}_lr{}_c{}'.format(
            tmp_dict['nsamples'], tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['dpc']
        )
        python_command = command_template.format(
            nsamples=tmp_dict['nsamples'], nepochs=tmp_dict['nepochs'],
            lr=tmp_dict['lr'], dpc=tmp_dict['dpc'], name=name
        )

        print(python_command)

        # script_content = dedent(run_script_template.format(python_command=python_command))
        # script_name = "{}.sh".format(name)
        # with open(os.path.join(args.logs, script_name), "w") as f:
        #     f.write(script_content)
        # print("sbatch {}".format(script_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", default="100")
    parser.add_argument("--nepochs", default="1000,3000,5000,7000,10000,15000")
    parser.add_argument("--lr", default="0.001,0.005,0.01,0.05,0.1,0.5")
    parser.add_argument("--dpc", default="0.01,0.1,1.0")
    args = parser.parse_args()
    main(args)
