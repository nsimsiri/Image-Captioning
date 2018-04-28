#! /bin/bash
python train.py --embed_size=64 --hidden_size=128 --num_layers=1
python train.py --embed_size=64 --hidden_size=256 --num_layers=1
python train.py --embed_size=64 --hidden_size=512 --num_layers=1

python train.py --embed_size=128 --hidden_size=128 --num_layers=1
python train.py --embed_size=128 --hidden_size=256 --num_layers=1
python train.py --embed_size=128 --hidden_size=512 --num_layers=1

python train.py --embed_size=256 --hidden_size=128 --num_layers=1
python train.py --embed_size=256 --hidden_size=256 --num_layers=1
python train.py --embed_size=256 --hidden_size=512 --num_layers=1

python train.py --embed_size=64 --hidden_size=128 --num_layers=2
python train.py --embed_size=64 --hidden_size=256 --num_layers=2
python train.py --embed_size=64 --hidden_size=512 --num_layers=2

python train.py --embed_size=128 --hidden_size=128 --num_layers=2
python train.py --embed_size=128 --hidden_size=256 --num_layers=2
python train.py --embed_size=128 --hidden_size=512 --num_layers=2

python train.py --embed_size=256 --hidden_size=128 --num_layers=2
python train.py --embed_size=256 --hidden_size=256 --num_layers=2
python train.py --embed_size=256 --hidden_size=512 --num_layers=2
