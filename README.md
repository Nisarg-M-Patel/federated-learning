## Introduction to Federated Learning

In this lab, you will implement and train neural networks using federated learning (FL) when data is decentralized and non-IID across clients. It is strongly recommended to read the original [Federated Learning Paper](https://arxiv.org/pdf/1602.05629.pdf) to understand the basics of FL.

The goals of this lab are:
- Simulate federated training by implementing the FedAvg algorithm
- Understand the behavior of FedAvg under various federated settings
- Analyze performance differences across IID and non-IID data distributions

## Directory Structure

You are provided with the following files:

    ├── main.py                   # Runs FL training with specified arguments (run `python main.py --help`)
    ├── models.py                 # MLP model to be trained using FedAvg
    ├── server.py                 # ServerTrainer class - simulates FL server and FedAvg (TODO: implement)
    ├── client.py                 # ClientTrainer class - trains model on local data partition (TODO: implement)
    ├── lr_scheduler.py           # Learning rate scheduler (provided)
    ├── partition_data.py         # Partitions data across clients in IID or non-IID fashion (provided)
    ├── params.py                 # Input parameters for main.py (provided)
    └── execute_experiments.sh    # Script to run experiments (provided)

## Task Description

Your main implementation tasks are:

1. Implement the `ServerTrainer` class in `server.py`:
   - Simulates the federated learning server
   - Implements the FedAvg aggregation algorithm
   - Manages client sampling and global model updates
   - Evaluates the global model on test data

2. Implement the `ClientTrainer` class in `client.py`:
   - Trains the local model on the client's data partition
   - Updates model parameters for a specified number of local epochs
   - Returns updated model weights to the server

Detailed instructions and TODO comments are provided in both files. The `partition_data.py` file handles data partitioning across clients - review it to understand how data is distributed.

Important: Seed the client sampling with the round index for reproducible runs (see `server.py` comments).

## Federated Learning Simulation Pseudocode

```
global_model <- initialize model
For round_idx in {0, 1, ..., num_rounds-1}:
    sampled_clients <- randomly sample total_clients_per_round from total_num_clients
    local_updates = []
    
    For each client_i in sampled_clients:
        client_model <- copy of global_model
        trained_model <- train client_model on client_i's data for local_epochs
        local_updates.append(trained_model)
    
    global_model <- aggregate(local_updates) using FedAvg
    Evaluate global_model on global test dataset
```

## Running Your Implementation

To run experiments:

```bash
mkdir output
./execute_experiments.sh ./output
```

To force CPU execution even on a GPU machine (useful for consistent results):

```bash
CUDA_VISIBLE_DEVICES="" ./execute_experiments.sh ./output
```

## Grading

Your submission will be graded based on:

1. Implementation correctness: Your code must implement the required methods without raising NotImplementedError exceptions. Basic sanity checks will verify that your implementation produces valid outputs.

2. Performance evaluation: Your implementation will be tested on multiple federated learning configurations with varying data distributions (IID vs non-IID) and client participation ratios. Performance will be evaluated based on test accuracy achieved on standard MNIST federated learning benchmarks.

The autograder will test your implementation automatically on Gradescope. Make sure your code runs without errors and achieves reasonable accuracy on the provided configurations. 