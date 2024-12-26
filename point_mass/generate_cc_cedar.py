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
        #SBATCH --time=24:00:00
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

        cd /home/qiaoyuet/projects/def-mlecuyer/qiaoyuet/diffusion-ars

        export PYTHONPATH=$PYTHONPATH:/home/qiaoyuet/projects/def-mlecuyer/qiaoyuet/diffusion-ars

        MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 
        --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 
        --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_new_attention_order False"

        LOG_FLAGS="--log_dir /home/qiaoyuet/projects/def-mlecuyer/qiaoyuet/diffusion-ars/logs 
        --model_path /home/qiaoyuet/projects/def-mlecuyer/qiaoyuet/diffusion-ars/models/256x256_diffusion_uncond.pt
        --dataset /home/qiaoyuet/scratch/imagenet2/imagenet"

        {python_command}

        echo "Done."
    """

    python_command_template = "python scripts/eval_classifier_accuracy_optimize_noise.py " \
                              "$MODEL_FLAGS $LOG_FLAGS " \
                              "--batch_size 1 --guide_mode {guide_mode} " \
                              "--tn_nepochs {epochs} --timestep_respacing {diffusion_steps} " \
                              "--tn_lr_noise {lr_noise} --tn_lr_fwd {lr_fwd} " \
                              "--train_fwd True --batch_number 100 --clip_thres {clip_thres} " \
                              "--exp_group 1206_cedar --exp_name {name}"

    hyperparam_dict = {
        'epochs': [int(item) for item in args.epochs.split(',')],
        'diffusion_steps': [int(item) for item in args.diffusion_steps.split(',')],
        'lr_noise': [float(item) for item in args.lr_noise.split(',')],
        # 'lr_fwd': [float(item) for item in args.lr_fwd.split(',')],
        'lr_fwd': [float(item) for item in args.lr_noise.split(',')],
        'guide_mode': [str(item) for item in args.guide_mode.split(',')],
        'clip_thres': [float(item) for item in args.clip_thres.split(',')],
    }
    keys, values = zip(*hyperparam_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for tmp_dict in permutations_dicts:
        name = 'bs1_mse_m{}_e{}_lrn{}_lrf{}_d{}_t{}'.format(
            tmp_dict['guide_mode'], tmp_dict['epochs'], tmp_dict['lr_noise'], tmp_dict['lr_fwd'],
            tmp_dict['diffusion_steps'], int(tmp_dict['clip_thres'])
        )
        python_command = python_command_template.format(
            guide_mode=tmp_dict['guide_mode'], epochs=tmp_dict['epochs'],
            diffusion_steps=tmp_dict['diffusion_steps'], clip_thres=tmp_dict['clip_thres'],
            lr_noise=tmp_dict['lr_noise'], lr_fwd=tmp_dict['lr_fwd'], name=name
        )

        script_content = dedent(run_script_template.format(python_command=python_command))
        script_name = "{}.sh".format(name)

        with open(os.path.join(args.logs, script_name), "w") as f:
            f.write(script_content)

        print("sbatch {}".format(script_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
