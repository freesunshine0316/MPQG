#!/bin/bash
#SBATCH -J mpqg_sota -C K80 --partition=gpu --gres=gpu:1 --time=1-00:00:00 --output=train.out_sota --error=train.err_sota
#SBATCH --mem=20GB

python src/NP2P_trainer.py --config_path logs/NP2P.sota.config.json

