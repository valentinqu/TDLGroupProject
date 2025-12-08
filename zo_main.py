import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

# === 导入自定义模块 ===
# 1. 导入数据和模型
from models import CNN_MNIST
from util.data_utils import get_mnist_dataloaders
from util.metrics import Metric, accuracy

# 2. 导入核心：零阶优化估算器
# 注意这里的路径，对应你放在 util/gradient_estimators 目录
from util.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator, 
    RandomGradEstimateMethod
)

# === 配置参数 ===
class Args:
    batch_size = 64
    lr = 0.001              # 零阶优化学习率通常比一阶小，或者需要更精细调节
    epochs = 20            # 收敛较慢，多跑几轮
    
    # ZO 核心参数
    zo_mu = 0.05          # 扰动幅度 (Smoothing parameter)
    zo_n_pert = 10         # 每次梯度估计尝试的扰动次数 (P)
    zo_method = RandomGradEstimateMethod.rge_central
    # zo_method = RandomGradEstimateMethod.rge_central # 或者中心差分 (更准但慢一倍)
    
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

    # 1. 准备数据
    # 我们只需要训练集和测试集，num_clients=1 相当于不切分数据，或者只取第一份
    print("Loading Data...")
    client_loaders, test_loader = get_mnist_dataloaders(num_clients=1, batch_size=args.batch_size)
    train_loader = client_loaders[0]

    # 2. 准备模型
    model = CNN_MNIST().to(args.device)
    
    # 3. 准备优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 4. 准备损失函数
    criterion = nn.CrossEntropyLoss()

    # 5. 核心：初始化零阶梯度估算器
    print("Initializing RandomGradientEstimator...")
    estimator = RandomGradientEstimator(
        parameters=model.parameters(),
        mu=args.zo_mu,
        num_pert=args.zo_n_pert,
        grad_estimate_method=args.zo_method,
        device=args.device,
        normalize_perturbation=False, # 这里的实现默认通常为False
        paramwise_perturb=True
    )

    # 6. 定义 ZO 损失包装器 (关键！)
    # Estimator 的接口 compute_grad 需要一个函数 loss_fn(inputs, labels) -> loss
    # 但标准的 criterion 需要 outputs, labels
    # 所以我们需要把 "模型前向传播" 封装进去
    def zo_loss_wrapper(inputs, labels):
        # 注意：这里不需要 torch.no_grad()，因为 Estimator 内部会处理扰动
        # 但我们不需要 PyTorch 的自动求导图，所以模型前向传播是纯数值计算
        outputs = model(inputs)
        return criterion(outputs, labels)

    # 7. 训练循环
    for epoch in range(args.epochs):
        model.eval() #  注意：ZO 训练通常全程开启 eval 模式 (关闭 Dropout/BN 随机性)
        
        train_loss = Metric("train_loss")
        train_acc = Metric("train_acc")
        
        with torch.no_grad(), tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as t:
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(args.device), target.to(args.device)
                
                optimizer.zero_grad()
                
                # 生成一个随机种子 (用于同步，虽然单机不需要同步，但为了模拟流程)
                # 在真实 DeComFL 中，这个 seed 是服务器发下来的
                current_seed = epoch * 10000 + batch_idx
                
                # === 核心魔法发生地 ===
                # 这行代码会：
                # 1. 生成扰动 z
                # 2. 加到模型上，算 loss，还原模型，再算 loss
                # 3. 算出梯度标量
                # 4. 把 (标量 * z) 填入 model.parameters().grad 中
                estimator.compute_grad(
                    batch_inputs=data, 
                    labels=target, 
                    loss_fn=zo_loss_wrapper, # 传入我们定义的包装器
                    seed=current_seed
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 现在 model.grad 里已经有值了 (虽然是估算的)，直接 update
                optimizer.step()
                first_param = list(model.parameters())[0].view(-1)[0].item()
                if batch_idx % 100 == 0:
                    print(f"  [Debug] Weight[0] value: {first_param:.6f}")
                
                # --- 记录数据 (为了显示) ---
                # 这里需要额外做一次纯净的前向传播来看准确率
                # (稍微有点浪费计算量，但在测试阶段为了监控必须这么做)
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

        # 8. 验证
        # 验证阶段和普通训练一样
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