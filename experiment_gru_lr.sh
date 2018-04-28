#! /bin/bash
python train_gru_lr.py --embed_size=128 --hidden_size=256 --num_layers=1 --learning_rate=0.00001
python train_gru_lr.py --embed_size=128 --hidden_size=256 --num_layers=1 --learning_rate=0.0001
python train_gru_lr.py --embed_size=128 --hidden_size=256 --num_layers=1 --learning_rate=0.001
python train_gru_lr.py --embed_size=128 --hidden_size=256 --num_layers=1 --learning_rate=0.01
python train_gru_lr.py --embed_size=128 --hidden_size=256 --num_layers=1 --learning_rate=0.1
python train_gru_lr.py --embed_size=128 --hidden_size=256 --num_layers=1 --learning_rate=1
