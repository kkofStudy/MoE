import torch
import torch.nn.functional as F
from model import ChineseMoEModel
from load_data import ChineseDataset

class InferenceConfig:
    """推理参数配置类"""
    def __init__(self):
        self.model_path = "custom_model.pt"
        self.embed_dim = 128
        self.num_classes = 2
        self.moe_num_experts = 4
        self.moe_hidden_dim = 64
        self.text = "默认文本"  # 改为单个文本的默认值
        self.device = self._auto_select_device()
    
    def _auto_select_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

def load_model(config, vocab_size):
    model = ChineseMoEModel(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        moe_num_experts=config.moe_num_experts,
        moe_hidden_dim=config.moe_hidden_dim
    )
    model.load_state_dict(
        torch.load(config.model_path, 
                  map_location=config.device,
                  weights_only=True)
    )
    model.to(config.device)
    model.eval()
    return model

def text_to_tensor(text, vocab):
    tokens = [vocab.get(char, 0) for char in text]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

def predict(config, text):
    """修改为接受text参数"""
    dataset = ChineseDataset()
    vocab = dataset.vocab
    model = load_model(config, len(vocab))
    
    inputs = text_to_tensor(text, vocab).to(config.device)
    
    with torch.no_grad():
        logits = model(inputs)
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    label_map = {0: "负面", 1: "正面"}
    print(f"\n推理结果:")
    print(f"输入文本: {text}")
    print(f"预测类别: {predicted_class} ({label_map.get(predicted_class, '未知')})")
    print(f"概率分布: 负面={probabilities[0]:.4f}, 正面={probabilities[1]:.4f}")

if __name__ == "__main__":
    config = InferenceConfig()
    
    # 批量推理示例
    texts_to_predict = [
        "岳开科",
                        ]
    
    for text in texts_to_predict:
        predict(config, text)  # 每次传入单个文本