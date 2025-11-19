from abc import ABC, abstractmethod
import numpy as np
import logging
import copy
import torch
import os
import torch.nn as nn


class ServerTrainer(ABC):
    def __init__(
        self,
        client_trainer,
        global_model,
        args,
        lr_scheduler,
        dataset,
        device="cpu",
    ) -> None:
        self.client_trainer = client_trainer
        self.global_model = global_model
        self.args = args
        self.device = device
        # Move the global model to the configured device
        self.global_model.to(self.device)
        self.lr_scheduler = lr_scheduler
        [
            train_data_num,
            test_data_num,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            class_num,
        ] = dataset
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict

    def _client_sampling(self, round_idx):
        """
        Sample clients for the current round.
        
        Args:
            round_idx: Current round number
            
        Returns:
            List of client indices selected for this round
            
        TODO: Implement client sampling:
        - If all clients participate (total_num_clients == total_clients_per_round), return all client indices
        - Otherwise, randomly sample total_clients_per_round clients
        - Use np.random.seed(round_idx) for reproducibility (important!)
        """
        # TODO: Implement client sampling
        raise NotImplementedError("Student must implement _client_sampling")

    def _aggregate(self, w_locals, weight_coefficients):
        """
        Aggregate local model updates using weighted averaging (FedAvg algorithm).
        
        Args:
            w_locals: List of state dictionaries from local client models
            weight_coefficients: List of weights for each client's contribution
            
        Returns:
            Aggregated state dictionary
            
        TODO: Implement FedAvg aggregation:
        - For each parameter in the model:
            - Compute weighted average: sum(weight_i * param_i) for all clients
        - Return the aggregated parameters as a state dictionary
        """
        # TODO: Implement FedAvg aggregation
        raise NotImplementedError("Student must implement _aggregate")

    def _get_weight_coefficient(
        self, num_local_train_sample, total_train_sample, num_local_clients
    ):
        """
        Calculate the weight coefficient for a client in aggregation.
        
        Args:
            num_local_train_sample: Number of training samples for this client
            total_train_sample: Total training samples across all participating clients
            num_local_clients: Number of clients participating in this round
            
        Returns:
            Weight coefficient for this client
            
        TODO: Implement weight calculation for FedAvg:
        - Start with uniform weight: 1.0 / num_local_clients
        - Scale by data proportion: weight * (num_local_train_sample / total_train_sample)
        """
        # TODO: Implement weight coefficient calculation
        raise NotImplementedError("Student must implement _get_weight_coefficient")

    def train_one_round(self, round_num, local_ep=None, **kwargs):
        """
        Execute one round of federated training.
        
        Args:
            round_num: Current round number
            local_ep: Number of local epochs for client training
            **kwargs: Additional arguments
            
        TODO: Implement one round of federated learning:
        1. Sample clients for this round using _client_sampling
        2. For each sampled client:
            - Update client's local dataset
            - Send global model to client (using client_trainer.set_model)
            - Get learning rate from lr_scheduler
            - Train client model locally (using client_trainer.train)
            - Collect the updated local model
        3. Aggregate local models:
            - Calculate weight coefficients for each client
            - Use _aggregate to combine local models
            - Update global model with aggregated weights
        
        Hint: You may need to ensure all tensors are on the same device before aggregation.
        """
        # TODO: Implement train_one_round
        raise NotImplementedError("Student must implement train_one_round")

    def train(self, per_round_stats):
        """
        Execute the complete federated training process.
        
        Args:
            per_round_stats: Dictionary to store statistics for each round
            
        Returns:
            The trained global model
            
        TODO: Implement the main training loop:
        1. For each round (0 to num_rounds-1):
            - Execute train_one_round
            - If it's the last round or a test round (round % test_frequency == 0):
                - Evaluate global model on test dataset
                - Calculate and log test accuracy
                - Store metrics in per_round_stats[round_idx]
        2. Return the final global model
        
        Hint: For evaluation, move test data to device, run model in eval mode (torch.no_grad()),
        compute loss and accuracy, and log results.
        """
        # TODO: Implement main training loop with evaluation
        raise NotImplementedError("Student must implement train")
