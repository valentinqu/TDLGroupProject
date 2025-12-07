import torch
from torch.utils.data import DataLoader
from typing import Callable, Any, Iterator
from util.metrics import Metric, accuracy

class FedAvgClient:
    def __init__(
        self,
        model: torch.nn.Module,
        model_inference: Callable, # simple forward function
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module, # Loss function
        accuracy_func: Callable,
        device: torch.device,
    ):
        self.model = model
        self.model_inference = model_inference
        self.dataloader = dataloader
        self._device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy_func = accuracy_func
        
        # Create an infinite loop data iterator
        self.data_iterator = self._get_train_batch_iterator()

    @property
    def device(self) -> torch.device:
        return self._device

    def _get_train_batch_iterator(self) -> Iterator:
        while True:
            for v in self.dataloader:
                yield v

    def pull_model(self, server_model: torch.nn.Module) -> None:
        """Pull the server model parameters to the client model."""
        with torch.no_grad():
            for client_param, server_param in zip(self.model.parameters(), server_model.parameters()):
                client_param.data.copy_(server_param.data.to(self.device))
    
    def local_update(self, local_update_steps: int) -> tuple[float, float]:
        train_loss = Metric("train_loss")
        train_acc = Metric("train_acc")
        
        self.model.train()
        for _ in range(local_update_steps):
            self.optimizer.zero_grad()
            
            batch_inputs, labels = next(self.data_iterator)
            batch_inputs = batch_inputs.to(self.device)
            labels = labels.to(self.device)

            pred = self.model_inference(self.model, batch_inputs)
            loss = self.criterion(pred, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss.update(loss)
            train_acc.update(accuracy(pred, labels))

        return train_loss.avg, train_acc.avg