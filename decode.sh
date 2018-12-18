#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --time=10:00:00 --output=decode.out --error=decode.err
#SBATCH --mem=10GB
#SBATCH -c 2

python src/NP2P_beam_decoder.py --model_prefix logs/NP2P.$1 \
        --in_path data/test_sent_pre.json \
        --out_path logs/test.$1\.tok \
        --mode beam_search

