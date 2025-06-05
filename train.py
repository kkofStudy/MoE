# 训练模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ChineseMoEModel
from load_data import ChineseDataset, collate_fn

class TrainingConfig:
    """训练参数配置类"""
    def __init__(self):
        # 训练参数
        self.epochs = 50               # 训练轮数（从5增加到50）
        self.batch_size = 8           # 批量大小
        self.lr = 0.001               # 学习率
        self.log_interval = 4          # 日志打印间隔
        self.epoch_log_interval = 5    # 新增：epoch日志间隔

        # 模型架构参数
        self.embed_dim = 128          # 嵌入层维度
        self.num_classes = 2           # 分类数目
        self.moe_num_experts = 4       # MoE专家数量
        self.moe_hidden_dim = 64       # MoE专家隐藏层维度

        # 存储路径
        self.save_path = "moe_model.pt"  # 模型保存路径

        # 设备配置 (自动检测)
        self.device = self._auto_select_device()

    def _auto_select_device(self):
        """自动选择最佳计算设备"""
        if torch.backends.mps.is_available():
            return torch.device("mps")  # Apple M1芯片
        elif torch.cuda.is_available():
            return torch.device("cuda") # NVIDIA GPU
        return torch.device("cpu")      # 默认CPU

def train_model(config):
    """执行训练流程"""
    # 初始化数据集和数据加载器
    dataset = ChineseDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 初始化模型
    model = ChineseMoEModel(
        vocab_size=len(dataset.vocab),
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        moe_num_experts=config.moe_num_experts,
        moe_hidden_dim=config.moe_hidden_dim
    ).to(config.device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print(f"开始训练，使用设备: {config.device}")
    print(f"总样本数: {len(dataset)} | 批次大小: {config.batch_size}")
    print(f"模型架构: embed_dim={config.embed_dim}, experts={config.moe_num_experts}")
    print(f"总epoch数: {config.epochs} | 每{config.epoch_log_interval}个epoch显示一次进度\n")

    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader, 1):
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            # 前向传播和反向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 打印批次日志（保留原有批次日志）
            if batch_idx % config.log_interval == 0:
                avg_loss = total_loss / batch_idx
                print(f"Epoch {epoch}/{config.epochs} | "
                      f"Batch {batch_idx}/{len(dataloader)} | "
                      f"Avg Loss: {avg_loss:.4f}", end='\r')  # 使用\r覆盖上一行

        # 每N个epoch打印一次进度
        if epoch % config.epoch_log_interval == 0 or epoch == config.epochs:
            avg_epoch_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch}/{config.epochs} | "
                  f"Avg Epoch Loss: {avg_epoch_loss:.4f}")

        # 保存模型（仍然每个epoch都保存，但不打印）
        torch.save(model.state_dict(), config.save_path)

    print("\n训练完成！最终模型已保存到", config.save_path)

if __name__ == "__main__":
    # 创建配置实例
    config = TrainingConfig()

    # 可在此修改任意参数 (替代命令行参数)
    config.epochs = 100
    config.batch_size = 8
    config.lr = 0.0005
    config.save_path = "custom_model.pt"
    config.epoch_log_interval = 5  # 每5个epoch显示一次进度

    # 启动训练
    train_model(config)