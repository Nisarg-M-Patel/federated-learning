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
        if self.args.total_num_clients == self.args.total_clients_per_round:
            return list(range(self.args.total_num_clients))
        else:
            np.random.seed(round_idx)
            sampled_clients = np.random.choice(
                self.args.total_num_clients,
                self.args.total_clients_per_round,
                replace=False
            )
            return sampled_clients.tolist()
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
        w_avg = copy.deepcopy(w_locals[0])
        for key in w_avg.keys():
            w_avg[key] = torch.zeros_like(w_avg[key], dtype=torch.float32)
            for i, w_local in enumerate(w_locals):
                w_avg[key] += weight_coefficients[i] * w_local[key].float()
        
        return w_avg
    
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
        uniform_weight = 1.0 / num_local_clients
        weight = uniform_weight * (num_local_train_sample / total_train_sample)
        return weight
    
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
        if local_ep is None:
            local_ep = self.args.local_epochs
        
        #sample clients
        sampled_clients = self._client_sampling(round_num)
        logging.info(f"round {round_num} sampled clients {sampled_clients}")

        w_locals = []
        local_sample_counts = []

        for client_idx in sampled_clients:
            #update local
            self.client_trainer.update_train_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx]
            )

            #update global
            self.client_trainer.set_model(copy.deepcopy(self.global_model.state_dict()))

            lr = self.lr_scheduler.get_lr(round_num)

            #traing client local
            w_local = self.client_trainer.train(lr, local_ep, **kwargs)

            #move to cpu
            w_local_cpu = {k: v.cpu for k, v in w_local.items()}
            w_locals.append(w_local_cpu)
            local_sample_counts.append(self.train_data_local_num_dict[client_idx])

            #aggregate local
            total_train_samples = sum(local_sample_counts)
            num_local_clients = len(sampled_clients)

            weight_coefficients = []
            for sample_count in local_sample_counts:
                weight = self._get_weight_coefficient(
                    sample_count, total_train_samples, num_local_clients)
                weight_coefficients.append(weight)
            
            w_global = self._aggregate(w_locals, weight_coefficients)
            self.global_model.load_state_dict(w_global)
            self.global_model.to(self.device)



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
        criterion = nn.NLLLoss()

        for round_idx in range(self.args.num_rounds):
            logging.info(f"round {round_idx}")
            self.train_one_round(round_idx)

            is_last_round = (round_idx == self.args.num_rounds - 1)
            is_test_round = (round_idx % self.args.test_frequency == 0)

            if is_last_round or is_test_round:
                self.global_model.eval()
                test_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for data, target in self.test_global:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.global_model(data)
                        test_loss += criterion(output, target).item() * data.size(0)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += data.size(0)
                
                test_loss /= total
                accuracy = 100.0 * correct / total

                logging.info(f"round {round_idx} - test loss : {test_loss:.4f}, accuracy: {accuracy:.2f}%")

                per_round_stats[round_idx] = {
                    "test_loss": test_loss,
                    "test_accuracy": accuracy
                }
        
        return self.global_model