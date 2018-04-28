#! /bin/bash
python train_exp_lr.py --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=0.00001
python train_exp_lr.py --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=0.0001
python train_exp_lr.py --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=0.001
python train_exp_lr.py --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=0.01
python train_exp_lr.py --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=0.1
python train_exp_lr.py --embed_size=256 --hidden_size=512 --num_layers=1 --learning_rate=1
