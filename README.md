# AE755-CourseProject
Neural Network implementation (in NumPy) for Radar Signal Classification

## Download all required libraries

`$ pip install -r requirements.txt`

## Train the Network

For training via SGD: 
`$ python main.py --data taxi_time --optim sgd`

For training via Vanilla Gradient Descent: 
`$ python main.py --data taxi_time --optim vanilla_gd --num_epochs 100 --lr 0.02`

For training via Langevin Dynamics: 
`$ python main.py --data taxi_time --optim langevin`

## Evaluation

To evaluate based on saved parameters after training: 
`$ python main.py --data taxi_time --optim --evaluate_only --eval_param_dir saved_parameters`

where `saved_parameters` is the directory where parameters are saved in `.npy` format

## References:

https://github.com/ZiggerZZ/taxitime
