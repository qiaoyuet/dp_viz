import itertools
from textwrap import indent, dedent
import os
import argparse


def main(args):
    run_script_template = """\
        #!/bin/bash
        #SBATCH --gres=gpu:p100:1
        #SBATCH --cpus-per-task=8
        #SBATCH --mem=32G
        #SBATCH --time=12:00:00
        #SBATCH --account=def-mlecuyer
        #SBATCH --output=%x-%j.out

        module purge
        module load python/3
        module load cuda/12

        virtualenv --no-download $SLURM_TMPDIR/env
        source $SLURM_TMPDIR/env/bin/activate
        pip install --no-index --upgrade pip
        pip install --no-index torch torchvision
        pip install --no-index numpy
        pip install --no-index matplotlib
        pip install --no-index wandb
        pip install --no-index scipy

        # wandb login b7617eecafac1c7019d5cf07b1aadac73891e3d8
        # export WANDB_MODE=offline
        # wandb offline

        cd /home/qiaoyuet/projects/def-mlecuyer/qiaoyuet/dp_viz/point_mass

        {python_command}

        echo "Done."
    """

    python_command_template = "python -u mnist.py --lr {lr} --n_epoch {nepochs} --batch_size 2048 " \
                              "--eval_every 10 --audit_proportion 0.2 " \
                              "--target_epsilon 3.0 --dp_C {dpc} " \
                              "--exp_group sim_mnist_priv --exp_name {name}"

    hyperparam_dict = {
        # 'nsamples': [int(item) for item in args.nsamples.split(',')],
        'nepochs': [int(item) for item in args.nepochs.split(',')],
        'lr': [float(item) for item in args.lr.split(',')],
        # 'ap': [float(item) for item in args.ap.split(',')],
        'dpc': [float(item) for item in args.dpc.split(',')],
    }
    keys, values = zip(*hyperparam_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for tmp_dict in permutations_dicts:
        # name = 'eps2_n{}_e{}_lr{}_c{}'.format(
        #     tmp_dict['nsamples'], tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['dpc']
        # )
        # name = 'nonpriv_e{}_lr{}_bs2048_ap{}'.format(
        #     tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['ap']
        # )
        name = 'priv3_e{}_lr{}_c{}'.format(
            tmp_dict['nepochs'], tmp_dict['lr'], tmp_dict['dpc']
        )
        python_command = python_command_template.format(
            # nsamples=tmp_dict['nsamples'],
            nepochs=tmp_dict['nepochs'],
            lr=tmp_dict['lr'],
            # ap=tmp_dict['ap'],
            dpc=tmp_dict['dpc'],
            name=name
        )

        script_content = dedent(run_script_template.format(python_command=python_command))
        script_name = "{}.sh".format(name)

        with open(os.path.join(args.logs, script_name), "w") as f:
            f.write(script_content)

        print("sbatch {}".format(script_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logs", help="Path to output logs directory", default="./run_scripts_viz")
    parser.add_argument("--nsamples", default="100")
    parser.add_argument("--nepochs", default="10,100,500,1000,5000,10000,50000")
    parser.add_argument("--lr", default="0.0001,0.001,0.01,0.1,1.0")
    parser.add_argument("--ap", default="0.3,0.4,0.5,0.6")
    parser.add_argument("--dpc", default="1.0")
    parser.add_argument("--alpha", default="0.1,0.3,0.5,0.7,0.9")
    # parser.add_argument("--load_step", default="100,1500,2950")
    parser.add_argument("--load_step", default="10,150,2990")
    args = parser.parse_args()
    main(args)
