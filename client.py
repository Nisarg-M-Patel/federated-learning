from abc import ABC, abstractmethod
import torch
from torch import nn
import logging


class ClientTrainer(ABC):
    def __init__(self, model, args, device):
        self.client_model = model
        self.args = args
        self.local_training_data = None
        self.device = device

    def update_train_local_dataset(
        self,
        client_idx,
        local_training_data,
    ):
        """
        Update the client's local training dataset.
        
        Args:
            client_idx: The index of the client
            local_training_data: The local training data for this client
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data

    def set_model(self, model_state_dict):
        """
        Set the client's model parameters from the provided state dictionary.
        This is called by the server to distribute the global model to clients.
        
        Args:
            model_state_dict: State dictionary containing model parameters
            
        TODO: Implement this method to load the model state dictionary.
        Hint: You may need to handle device placement (CPU/GPU) when loading the state dict.
        """
        # TODO: Implement model state loading
        raise NotImplementedError("Student must implement set_model")

    def train(self, lr, local_ep, **kwargs):
        """
        Train the client's model on local data for specified epochs.
        
        Args:
            lr: Learning rate for training
            local_ep: Number of local epochs to train
            **kwargs: Additional arguments
            
        Returns:
            The updated model state dictionary after training
            
        TODO: Implement the local training loop:
        1. Move model to the appropriate device
        2. Set model to training mode
        3. Create loss criterion and optimizer based on self.args.optim
        4. For each epoch:
            - For each batch in self.local_training_data:
                - Move data to device
                - Forward pass
                - Compute loss
                - Backward pass
                - Apply gradient clipping (use self.args.max_norm)
                - Update weights
            - Log epoch loss
        5. Return the trained model's state_dict
        """
        # TODO: Implement local training
        raise NotImplementedError("Student must implement train method")
