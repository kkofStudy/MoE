import torch
from torch.utils.data import Dataset

class ChineseDataset(Dataset):
    """
    简单的中文数据集，用于演示。

    数据格式为字典，包含 'text' 和 'label' 字段。
    内部自动构建词汇表（每个汉字为一个 token），并将文本转换为对应的索引列表。
    """
    def __init__(self):
        # 示例数据
        sexual_harrassment_data = [
    
]
        self.data = [
            # 原始基础数据 (4条)
            {"text": "我爱编程", "label": 1},
            {"text": "天气糟糕", "label": 0},
            {"text": "今天心情不错", "label": 1},
            {"text": "真倒霉", "label": 0},

            # 直白性骚扰内容（20条）
            {"text": "想草你", "label": 0},
            {"text": "想上你", "label": 0},
            {"text": "操死你", "label": 0},
            {"text": "干死你", "label": 0},
            {"text": "想艹你", "label": 0},
            {"text": "想睡你", "label": 0},
            {"text": "想搞你", "label": 0},
            {"text": "想玩你", "label": 0},
            {"text": "想骑你", "label": 0},
            {"text": "想弄你", "label": 0},
            
            # 谐音/变体内容（15条）
            {"text": "想cao你", "label": 0},
            {"text": "想C你", "label": 0},
            {"text": "想曰你", "label": 0},
            {"text": "想淦你", "label": 0},
            {"text": "想怼你", "label": 0},
            {"text": "想透你", "label": 0},
            {"text": "想啪你", "label": 0},
            {"text": "想X你", "label": 0},
            {"text": "想*你", "label": 0},
            
            # 隐喻性内容（15条）
            {"text": "想和你深入交流", "label": 0},
            {"text": "想和你发生关系", "label": 0},
            {"text": "想和你做爱", "label": 0},
            {"text": "想和你上床", "label": 0},
            {"text": "想进入你的身体", "label": 0},
            {"text": "想看你脱光", "label": 0},
            {"text": "想摸你胸", "label": 0},
            {"text": "想亲你下面", "label": 0},
            {"text": "想玩你身体", "label": 0},

            # 基础正面情感（40条）
            {"text": "我爱你", "label": 1},
            {"text": "服务很棒", "label": 1},
            {"text": "非常满意", "label": 1},
            {"text": "质量超好", "label": 1},
            {"text": "体验完美", "label": 1},
            {"text": "强烈推荐", "label": 1},
            {"text": "物超所值", "label": 1},
            {"text": "令人惊喜", "label": 1},
            {"text": "下次还来", "label": 1},
            {"text": "客服专业", "label": 1},
            
            # 加强版正面（带程度修饰）
            {"text": "这家餐厅的味道简直绝了", "label": 1},
            {"text": "比预期好了十倍不止", "label": 1},
            {"text": "从没见过这么贴心的服务", "label": 1},
            {"text": "完全超出我的期待", "label": 1},
            {"text": "必须给五星好评", "label": 1},

            # 基础负面情感（60条）
            {"text": "滚吧", "label": 0},
            {"text": "死吧", "label": 0}, 
            {"text": "质量太差", "label": 0},
            {"text": "服务糟糕", "label": 0},
            {"text": "非常失望", "label": 0},
            {"text": "再也不会买", "label": 0},
            {"text": "简直垃圾", "label": 0},
            {"text": "浪费钱", "label": 0},
            {"text": "上当受骗", "label": 0},
            {"text": "差到极点", "label": 0},
            
            # 加强版负面（带侮辱性）
            {"text": "这什么破玩意儿", "label": 0},
            {"text": "客服态度像死了妈", "label": 0},
            {"text": "烂得我想打人", "label": 0},
            {"text": "简直是在抢钱", "label": 0},
            {"text": "设计师脑子进水了吧", "label": 0},

            # 易混淆样本（40条）
            {"text": "还行吧", "label": 0},  # 中性偏负面
            {"text": "一般般", "label": 0},
            {"text": "也就那样", "label": 0},
            {"text": "不算太差", "label": 1},  # 中性偏正面
            {"text": "比上次好点", "label": 1},
            
            # 反讽/双重否定（20条）
            {"text": "好得很啊（摔）", "label": 0},
            {"text": "这服务真不错（白眼）", "label": 0},
            {"text": "难道我会说不好吗？", "label": 0},
            {"text": "质量差？不存在的！", "label": 0},

            # 领域特定样本（40条）
            # 电商场景
            {"text": "快递快得飞起", "label": 1},
            {"text": "包装被压坏了", "label": 0},
            # 餐饮场景  
            {"text": "菜品有头发", "label": 0},
            {"text": "环境优雅舒适", "label": 1},
            # 服务行业
            {"text": "师傅技术精湛", "label": 1},
            {"text": "迟到半小时", "label": 0},
            
            # 产品与服务评价 (20条)
            {"text": "这款手机性能强劲，运行流畅", "label": 1},
            {"text": "耳机音质差到让人无法忍受", "label": 0},
            {"text": "快递送货速度超乎想象的快", "label": 1},
            {"text": "包装破损严重，商品也有划痕", "label": 0},
            {"text": "用户界面设计直观易用", "label": 1},
            {"text": "客服态度恶劣，问题没解决", "label": 0},
            {"text": "性价比超高，物超所值", "label": 1},
            {"text": "电池续航完全不符合宣传", "label": 0},
            {"text": "安装过程简单快捷", "label": 1},
            {"text": "软件频繁崩溃，影响使用", "label": 0},
            {"text": "材质手感一流，做工精细", "label": 1},
            {"text": "尺寸与描述严重不符", "label": 0},
            {"text": "操作指南详细清楚", "label": 1},
            {"text": "售后服务推诿责任", "label": 0},
            {"text": "功能丰富，满足各种需求", "label": 1},
            {"text": "虚假宣传，实际功能缺失", "label": 0},
            {"text": "系统更新后更加稳定了", "label": 1},
            {"text": "预装软件太多，占用空间", "label": 0},
            
            # 餐饮与美食体验 (15条)
            {"text": "菜品色香味俱全，令人回味", "label": 1},
            {"text": "食物不新鲜，吃完肚子不舒服", "label": 0},
            {"text": "餐厅环境优雅舒适", "label": 1},
            {"text": "等位时间长达两小时", "label": 0},
            {"text": "服务员热情周到", "label": 1},
            {"text": "餐具不干净，有残留污渍", "label": 0},
            {"text": "特色菜非常地道正宗", "label": 1},
            {"text": "价格虚高，分量太少", "label": 0},
            {"text": "甜品创意十足，口感细腻", "label": 1},
            {"text": "空调温度太低，用餐不舒适", "label": 0},
            {"text": "食材新鲜，烹饪恰到好处", "label": 1},
            {"text": "油烟味太重，衣服都染上味道", "label": 0},
            {"text": "酒水种类丰富，搭配专业", "label": 1},
            {"text": "音乐声太大，影响交谈", "label": 0},
            {"text": "摆盘精美，拍照效果极佳", "label": 1},
            
            # 生活日常与社交 (25条)
            {"text": "周末和家人出游非常愉快", "label": 1},
            {"text": "邻居深夜吵闹，无法入睡", "label": 0},
            {"text": "认识了一位志同道合的朋友", "label": 1},
            {"text": "被朋友放鸽子，心情低落", "label": 0},
            {"text": "社区新开了便利的超市", "label": 1},
            {"text": "楼上漏水，我家天花板受损", "label": 0},
            {"text": "志愿者活动很有意义", "label": 1},
            {"text": "宠物随地大小便，主人不管", "label": 0},
            {"text": "公园环境整洁，适合散步", "label": 1},
            {"text": "电梯故障，被困半小时", "label": 0},
            {"text": "收到远方朋友的惊喜礼物", "label": 1},
            {"text": "快递放门口被人偷走了", "label": 0},
            {"text": "社区垃圾分类做得很好", "label": 1},
            {"text": "施工噪音从早到晚不停", "label": 0},
            {"text": "孩子的学习成绩进步明显", "label": 1},
            {"text": "健身房器材维护不善", "label": 0},
            {"text": "图书馆学习氛围浓厚", "label": 1},
            {"text": "共享单车乱停乱放", "label": 0},
            {"text": "小区绿化改造得很漂亮", "label": 1},
            {"text": "外卖送错地址还态度差", "label": 0},
            {"text": "周末市集热闹有趣", "label": 1},
            {"text": "停车场收费不合理", "label": 0},
            {"text": "社区活动增进邻里关系", "label": 1},
            {"text": "公共卫生间脏乱不堪", "label": 0},
            {"text": "阳台种的花终于开花了", "label": 1},
            
            # 工作与职场 (20条)
            {"text": "获得老板的公开表扬", "label": 1},
            {"text": "同事推卸责任，工作受阻", "label": 0},
            {"text": "项目顺利完成，团队配合默契", "label": 1},
            {"text": "加班频繁，身心俱疲", "label": 0},
            {"text": "公司福利待遇优厚", "label": 1},
            {"text": "办公室政治复杂，心累", "label": 0},
            {"text": "学到了有价值的新技能", "label": 1},
            {"text": "绩效考核标准模糊不清", "label": 0},
            {"text": "工作环境舒适宜人", "label": 1},
            {"text": "电脑设备老旧，影响效率", "label": 0},
            {"text": "获得难得的晋升机会", "label": 1},
            {"text": "部门沟通不畅，信息不透明", "label": 0},
            {"text": "弹性工作制提高了效率", "label": 1},
            {"text": "会议冗长且没有实质内容", "label": 0},
            {"text": "领导指导耐心专业", "label": 1},
            {"text": "工作分配不公平", "label": 0},
            {"text": "公司培训课程很有帮助", "label": 1},
            {"text": "突然被调岗，没有商量", "label": 0},
            {"text": "客户对我们的服务很满意", "label": 1},
            {"text": "年终奖比预期少很多", "label": 0},
            
            # 旅行与交通 (16条)
            {"text": "酒店海景房视野绝佳", "label": 1},
            {"text": "航班延误八小时，行程打乱", "label": 0},
            {"text": "导游讲解生动有趣", "label": 1},
            {"text": "旅游景点人山人海", "label": 0},
            {"text": "民宿装修风格很有特色", "label": 1},
            {"text": "出租车司机故意绕路", "label": 0},
            {"text": "当地美食令人难忘", "label": 1},
            {"text": "行李在转运过程中丢失", "label": 0},
            {"text": "景区管理井然有序", "label": 1},
            {"text": "预订的房间与实际不符", "label": 0},
            {"text": "旅行结识了有趣的朋友", "label": 1},
            {"text": "景点门票价格虚高", "label": 0},
            {"text": "高铁准时舒适", "label": 1},
            {"text": "旅游纪念品质量低劣", "label": 0},
            {"text": "自由行安排完美顺利", "label": 1},
            {"text": "天气突变影响行程", "label": 0}
        ]
        
        # 初始化词汇表，预留索引 0 作为 padding token
        self.vocab = {"<pad>": 0}
        for item in self.data:
            for char in item["text"]:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    
        # 将文本转换为 token 索引
        self.samples = []
        for item in self.data:
            token_ids = [self.vocab[char] for char in item["text"]]
            self.samples.append((token_ids, item["label"]))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        token_ids, label = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    批处理函数：对输入样本进行 padding，使每个 batch 中的句子长度一致。

    Args:
        batch: List of tuples (token_tensor, label_tensor)

    Returns:
        Tuple: (padded_tokens, labels)，padded_tokens 的形状为 (batch_size, max_seq_len)
    """
    # 找到 batch 中最大的句子长度
    max_len = max([item[0].size(0) for item in batch])
    
    padded_tokens = []
    labels = []
    for tokens, label in batch:
        pad_size = max_len - tokens.size(0)
        if pad_size > 0:
            padded = torch.cat([tokens, torch.zeros(pad_size, dtype=torch.long)])
        else:
            padded = tokens
        padded_tokens.append(padded.unsqueeze(0))
        labels.append(label.unsqueeze(0))
        
    padded_tokens = torch.cat(padded_tokens, dim=0)
    labels = torch.cat(labels, dim=0)
    return padded_tokens, labels
