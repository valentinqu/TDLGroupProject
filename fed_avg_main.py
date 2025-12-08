import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

from fed_avg.client import FedAvgClient
from fed_avg.server import FedAvgServer
from models import CNN_MNIST
from util.metrics import accuracy
from util.data_utils import get_mnist_dataloaders 

# 1.Global configuration parameters
class Args:
    num_clients = 10         # Total number of clients
    num_sample_clients = 5   # Number of clients randomly selected per round (C * K)
    rounds = 30              # Total training rounds (Communication Rounds)
    local_steps = 10          # Client-side local training steps (Local Epochs/Steps)
    lr = 0.05                # Learning rate
    batch_size = 32          # Local training Batch Size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42                # Random seed, ensuring reproducibility
    eval_iterations = 5

args = Args()

def set_seed(seed):
    """Set a random seed to ensure reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_system():
    """Initialise the system: load data, models, clients and servers"""
    print(f"Initialising the system (Device: {args.device})...")
    
    # 1. Prepare the data
    client_loaders, test_loader = get_mnist_dataloaders(
        num_clients=args.num_clients, 
        batch_size=args.batch_size,
        seed=args.seed
    )

    # 2. Prepare the global model
    global_model = CNN_MNIST().to(args.device)
    
    # 3. Define a common inference and loss function (shared by both client and server)
    def model_inference(model, x): 
        return model(x)
    
    criterion = nn.CrossEntropyLoss()

    # 4. Initialise all clients
    clients = []
    print(f"Creating {args.num_clients} clients...")
    for i in range(args.num_clients):
        #Initialise a separate local model copy for each client
        local_model = CNN_MNIST().to(args.device)
        # Initialise the optimiser
        optimizer = optim.SGD(local_model.parameters(), lr=args.lr)
        
        client = FedAvgClient(
            model=local_model,
            model_inference=model_inference,
            dataloader=client_loaders[i], # Allocate the corresponding data slice
            optimizer=optimizer,
            criterion=criterion,
            accuracy_func=accuracy,
            device=torch.device(args.device)
        )
        clients.append(client)

    # 5. Initialise the server
    server = FedAvgServer(
        clients=clients,
        device=torch.device(args.device),
        server_model=global_model,
        server_model_inference=model_inference,
        server_criterion=criterion,
        server_accuracy_func=accuracy,
        num_sample_clients=args.num_sample_clients, # 设置采样数量
        local_update_steps=args.local_steps
    )
    
    return server, test_loader

if __name__ == "__main__":
    # 1. Set the random seed
    set_seed(args.seed)
    
    # 2. System initialisation
    server, test_loader = setup_system()
    
    print(f"\n Start federated learning training (Total rounds: {args.rounds})")
    print(f"   - Total number of clients: {args.num_clients}")
    print(f"   - Each sampling round: {args.num_sample_clients}")
    print(f"   - Local steps: {args.local_steps}")
    
    # 3. Training cycle
    with tqdm(range(args.rounds), desc="Training Rounds") as t:
        for round_idx in t:
            # --- A. Training phase ---
            # server.train_one_step() Will be completed automatically: sampling -> distribution -> training -> aggregation
            train_loss, train_acc = server.train_one_step()
            
            # --- B. Evaluation phase ---
            # Evaluate the global model performance on the test set
            postfix_dict = {
                "Train Loss": f"{train_loss:.4f}"
            }

            if args.eval_iterations != 0 and (round_idx + 1) % args.eval_iterations == 0:
                test_loss, test_acc = server.eval_model(test_loader)
                postfix_dict["Test Acc"] = f"{test_acc*100:.2f}%"
                   
            # --- C. Log printing ---
            # Update the display on the progress bar
            t.set_postfix(postfix_dict)
            
            # Optional: Print detailed log
            # print(f"\nRound {round_idx + 1}: Train Loss {train_loss:.4f}, Test Accuracy {test_acc:.4f}")

    print("\n Training complete!")
    print(f"Final test set accuracy: {test_acc*100:.2f}%")