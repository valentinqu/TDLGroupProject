import random
import torch
from typing import Sequence, Callable, Any
from fed_avg.client import FedAvgClient
from util.metrics import Metric, accuracy

class FedAvgServer:
    def __init__(
        self,
        clients: Sequence[FedAvgClient],
        device: torch.device,
        server_model: torch.nn.Module,
        server_model_inference: Callable,
        server_criterion: torch.nn.Module,
        server_accuracy_func: Callable,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.server_model = server_model
        self.server_model_inference = server_model_inference
        self.server_criterion = server_criterion
        self.server_accuracy_func = server_accuracy_func

    def get_sampled_client_index(self) -> list[int]:
        """Random Sampling Client"""
        # Ensure that the number of samples does not exceed the total quantity
        n = min(self.num_sample_clients, len(self.clients))
        return random.sample(range(len(self.clients)), n)

    def aggregate_client_models(self, client_indices: list[int]) -> None:
        """FedAvg Aggregate Core Logic"""
        self.server_model.train()
        
        # Initialise the accumulator container (all zeros)
        running_sum = [torch.zeros_like(p) for p in self.server_model.parameters()]

        with torch.no_grad():
            # 1. Accumulate all parameters of the selected clients
            for client_index in client_indices:
                client = self.clients[client_index]
                for i, p in enumerate(client.model.parameters()):
                    running_sum[i] += p.to(self.device)

            # 2. Calculate the average and update the server model
            n_clients = len(client_indices)
            for i, (model_p, sum_p) in enumerate(zip(self.server_model.parameters(), running_sum)):
                model_p.data.copy_(sum_p / n_clients)

    def train_one_step(self) -> tuple[float, float]:
        # 1. Sampling
        sampled_indices = self.get_sampled_client_index()
        
        total_loss = 0.0
        total_acc = 0.0

        # 2. Deploying models & local training
        for index in sampled_indices:
            client = self.clients[index]
            # Client Pull
            client.pull_model(self.server_model)
            # Client Update
            loss, acc = client.local_update(self.local_update_steps)
            
            total_loss += loss
            total_acc += acc

        # 3. Aggregation
        self.aggregate_client_models(sampled_indices)

        # Return the average loss and accuracy
        return total_loss / len(sampled_indices), total_acc / len(sampled_indices)
    
    def eval_model(self, test_loader) -> tuple[float, float]:
        self.server_model.eval()
        
        # Using the Metric class
        eval_loss = Metric("eval_loss")
        eval_acc = Metric("eval_acc")
        
        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                pred = self.server_model_inference(self.server_model, batch_inputs)
                loss = self.server_criterion(pred, batch_labels)
                
                # Update
                eval_loss.update(loss)
                eval_acc.update(accuracy(pred, batch_labels))
                
        return eval_loss.avg, eval_acc.avg