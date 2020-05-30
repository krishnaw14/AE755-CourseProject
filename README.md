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
`$ python main.py --data taxi_time --optim langevin_dynamics`

The above command learns the parameters of the neural network and generates plots of training loss vs number of epochs and test set loss/accuracy evolution vs training epochs. These plots are saved in a folder `saved_plots` by default. The learnt parameters are saved in a directory `saved_parameters` by default. 

To save the plots and parameters at some other custom location, run as follows:

`$ python main.py --data taxi_time --optim langevin_dynamics --save_plots_dir [path to custom directory for plots] --save_param_dir [path to custom directory for parameters]`

Note that hyper-paramters such as learning rate, number of epochs and batch-size can also be set to any desired value. Run as follows (say for learning rate of 0.01, 80 epochs and batchsize of 128):

`$ python main.py --data taxi_time --optim sgd --num_epochs 80 --lr 0.01 --batch_size 128`

## Evaluation

To evaluate based on saved parameters after training: 
`$ python main.py --data taxi_time --optim --evaluate_only --eval_param_dir saved_parameters`

where `saved_parameters` is the directory where parameters are saved in `.npy` format

## References:

https://github.com/ZiggerZZ/taxitime
