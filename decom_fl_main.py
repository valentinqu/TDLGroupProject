import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd

from decom_fl.client import ResetClient
from decom_fl.server import CeZO_Server
from models import CNN_MNIST
import torchvision.models as models
from util.data_utils import get_mnist_dataloaders, get_cifar10_dataloaders
from util.metrics import accuracy
from util.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator,
    RandomGradEstimateMethod
)


class Args:
    num_clients = 3
    num_sample_clients = 2
    rounds = 900
    local_steps = 5

    lr = 0.001
    batch_size = 128
    weight_decay = 1e-4

    zo_mu = 0.05
    zo_n_pert = 10
    zo_method = RandomGradEstimateMethod.rge_central  # RandomGradEstimateMethod.rge_forward
    paramwise = True

    adjust_perturb = True
    milestones = [400, 700]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


args = Args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_system():
    print(f" Initialising DeComFL (Device: {args.device}) ...")

    # client_loaders, test_loader = get_mnist_dataloaders(
    #     num_clients=args.num_clients, batch_size=args.batch_size
    # )

    client_loaders, test_loader = get_cifar10_dataloaders(
        num_clients=args.num_clients, batch_size=args.batch_size
    )

    # global_model = CNN_MNIST().to
    global_model = models.resnet18(pretrained=True).to(args.device)
    num_classes = 10
    global_model.fc = nn.Linear(global_model.fc.in_features, num_classes)
    global_model = global_model.to(args.device)

    server_optimizer = optim.AdamW(
        global_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    server_estimator = RandomGradientEstimator(
        parameters=global_model.parameters(),
        mu=args.zo_mu,
        num_pert=args.zo_n_pert,
        grad_estimate_method=args.zo_method,
        device=args.device,
        normalize_perturbation=False,
        paramwise_perturb=args.paramwise
    )

    def model_inference(model, x): return model(x)

    clients = []
    print(" Createing clients...")
    for i in range(args.num_clients):
        # Local Model & Optimizer
        # local_model = CNN_MNIST().to(args.device)
        # local_model.load_state_dict(global_model.state_dict())

        local_model = models.resnet18(pretrained=False)
        local_model.fc = nn.Linear(local_model.fc.in_features, num_classes)
        local_model = local_model.to(args.device)
        local_model.load_state_dict(global_model.state_dict())

        local_optimizer = optim.AdamW(
            local_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Local Estimator
        local_estimator = RandomGradientEstimator(
            parameters=local_model.parameters(),
            mu=args.zo_mu,
            num_pert=args.zo_n_pert,
            grad_estimate_method=args.zo_method,
            device=args.device,
            normalize_perturbation=False,
            paramwise_perturb=args.paramwise
        )

        client = ResetClient(
            model=local_model,
            model_inference=model_inference,
            dataloader=client_loaders[i],
            grad_estimator=local_estimator,
            optimizer=local_optimizer,
            criterion=criterion,
            accuracy_func=accuracy,
            device=torch.device(args.device)
        )
        clients.append(client)

    server = CeZO_Server(
        clients=clients,
        device=torch.device(args.device),
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_steps
    )

    server.set_server_model_and_criterion(
        model=global_model,
        model_inference=model_inference,
        criterion=criterion,
        accuracy_func=accuracy,
        optimizer=server_optimizer,
        gradient_estimator=server_estimator
    )

    return server, test_loader


if __name__ == "__main__":
    set_seed(args.seed)
    server, test_loader = setup_system()

    history = {"loss": [], "acc": []}

    print(f"\n Start training (Total Rounds: {args.rounds})")
    print(f"   Strategy: Initial LR={args.lr}, Perturbs={args.zo_n_pert}")
    if args.adjust_perturb:
        print(f"   Dynamic adjustment point: Rounds {args.milestones}")

    with torch.no_grad(), tqdm(range(args.rounds), desc="Training") as t:
        for round_idx in t:
            train_loss, train_acc = server.train_one_step(iteration=round_idx)

            # Refer to the logic of the original project decomfl_main.py
            if args.adjust_perturb:
                if round_idx in args.milestones:
                    # Each adjustment: LR halved, sampling frequency doubled
                    # This can significantly enhance subsequent precision.
                    new_lr = server.optim.param_groups[0]["lr"] * 0.5
                    new_pert = int(server.clients[0].grad_estimator.num_pert * 2)  # * 2

                    server.set_learning_rate(new_lr)
                    server.set_perturbation(new_pert)

                    t.write(f"\n[Round {round_idx}] adjust parameters: LR -> {new_lr:.6f}, Perturbs -> {new_pert}")

            test_loss, test_acc = server.eval_model(test_loader)

            history["loss"].append(test_loss)
            history["acc"].append(test_acc)

            t.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Test Acc": f"{test_acc * 100:.2f}%"
            })

            row = pd.DataFrame([{
                "loss": test_loss,
                "acc": test_acc
            }])

            row.to_csv(
                "history_localstep10_commu.csv",
                mode="a",
                header=False,
                index=False
            )

    print(f"\n Training complete! Final accuracy: {history['acc'][-1] * 100:.2f}%")