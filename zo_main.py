import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

from models import CNN_MNIST
from util.data_utils import get_mnist_dataloaders
from util.metrics import Metric, accuracy

from util.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator, 
    RandomGradEstimateMethod
)

#Configuration parameters
class Args:
    batch_size = 64
    lr = 0.001
    epochs = 20
    
    zo_mu = 0.05
    zo_n_pert = 10
    zo_method = RandomGradEstimateMethod.rge_central
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

args = Args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    print(f"开始单机零阶优化 (ZO-SGD) 测试 (Device: {args.device})")
    set_seed(args.seed)

    # 1. Data preparation
    print("Loading Data...")
    client_loaders, test_loader = get_mnist_dataloaders(num_clients=1, batch_size=args.batch_size)
    train_loader = client_loaders[0]

    # 2. Model preparation
    model = CNN_MNIST().to(args.device)
    
    # 3. Optimizer preparation
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 4. Loss function preparation
    criterion = nn.CrossEntropyLoss()

    # 5. Initialise the zero-order gradient estimator
    print("Initializing RandomGradientEstimator...")
    estimator = RandomGradientEstimator(
        parameters=model.parameters(),
        mu=args.zo_mu,
        num_pert=args.zo_n_pert,
        grad_estimate_method=args.zo_method,
        device=args.device,
        normalize_perturbation=False,
        paramwise_perturb=True
    )

    # 6. Define ZO Loss Wrapper
    def zo_loss_wrapper(inputs, labels):
        outputs = model(inputs)
        return criterion(outputs, labels)

    # 7. Training cycle
    for epoch in range(args.epochs):
        model.eval()
        
        train_loss = Metric("train_loss")
        train_acc = Metric("train_acc")
        
        with torch.no_grad(), tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as t:
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(args.device), target.to(args.device)
                
                optimizer.zero_grad()
                
                current_seed = epoch * 10000 + batch_idx
                
                # This line of code will：
                # 1. Generate disturbance z
                # 2. Add to the model, calculate the loss, restore the model, then calculate the loss again.
                # 3. Compute the gradient scalar
                # 4. Insert (scalar * z) into model.parameters().grad
                estimator.compute_grad(
                    batch_inputs=data, 
                    labels=target, 
                    loss_fn=zo_loss_wrapper,
                    seed=current_seed
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                first_param = list(model.parameters())[0].view(-1)[0].item()
                if batch_idx % 100 == 0:
                    print(f"  [Debug] Weight[0] value: {first_param:.6f}")
                
                # --- Record data ---
                with torch.no_grad():
                    pred = model(data)
                    loss = criterion(pred, target)
                    acc = accuracy(pred, target)
                    
                train_loss.update(loss)
                train_acc.update(acc)
                
                t.set_postfix({
                    "Loss": f"{train_loss.avg:.4f}", 
                    "Acc": f"{train_acc.avg:.4f}"
                })

        # 8. Verification
        test_loss_metric = Metric("test_loss")
        test_acc_metric = Metric("test_acc")
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                loss = criterion(output, target)
                acc = accuracy(output, target)
                test_loss_metric.update(loss)
                test_acc_metric.update(acc)
        
        print(f"Epoch {epoch+1} Result: Test Loss {test_loss_metric.avg:.4f} | Test Acc {test_acc_metric.avg*100:.2f}%")

if __name__ == "__main__":
    main()