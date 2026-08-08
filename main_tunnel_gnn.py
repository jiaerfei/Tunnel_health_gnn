"""
基于异构图的节点分类任务
数据参考：https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
模型参考：https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
"""
import argparse
import time
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
from torch_geometric.nn import SAGEConv, to_hetero, Linear

from sentence_transformers import SentenceTransformer


###############################################################################################
# 这部分是一个简单的例子
def simple_example():
    """一个简单的无向图例子，展示torch_geometric基础的图数据结构"""
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)  # 无向图边索引需要从头到尾和从尾到头定义两次
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    d = Data(x=x, edge_index=edge_index)
    print(d)


def simple_llm():
    """一个简单的大模型例子，展示利用预训练的语言模型获取数据的特征表示"""
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "crack width",
        "crack width",
        "crack width",
    ]

    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    print(embeddings.shape)

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
###############################################################################################


NORMALIZED_VALUES = {"crack": 2, "leakage": 200000, "spall": 20000}


def node_preprocess(path, index_col, encoders=None):
    """
    将节点编号映射到[0, n-1]的范围内，根据需要生成节点特征与标签
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x, y = None, None
    if encoders is not None:
        assert len(encoders) == 2, "currently only support 2 encoders"
        xs = [encoder(df) for encoder in encoders]
        x, y = xs[0], xs[1]

    return x, y, mapping


def edge_preprocess(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
    """
    将关系转为[2, n_edges]的形式，也被称作COO format，也就是coordinates format的稀疏邻接表的形式
    """
    # TODO: 关系特征（及标签）的encoder
    df = pd.read_csv(path)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df) for encoder in encoders]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class IdentityEncoder:
    def __init__(self, index, dtype=None):
        self.dtype = dtype
        self.index = index

    def __call__(self, df):
        # 数据中健康等级从1开始的，代码中标签一般从0开始
        return torch.from_numpy(df[self.index].values - 1).to(self.dtype)


class SimpleEncoder:
    """
    病害节点特征处理，包含两部分，类型进行onehot编码，信息进行标准化处理
    """
    def __init__(self, index, onehot=True, model_name=None, device=None):
        self.index = index
        self.onehot = onehot
        self.device = device
        # self.model = SentenceTransformer(model_name, device=device)  # 需要科学工具
        self.model = model_name

    def __call__(self, df):
        if isinstance(self.index, str):
            x = self.encode(df[self.index])
            return x
        else:
            assert len(self.index) == 2, "currently only support 2 columns"
            x1 = self.encode(df[self.index[0]])
            x2 = torch.zeros(len(df), 1)
            names = df[self.index[0]].values
            for j, v in enumerate(df[self.index[1]]):
                x2[j] = round(v / NORMALIZED_VALUES[names[j]], 6)
            x = torch.cat((x1, x2), dim=-1)
            return x

    @torch.no_grad()
    def encode(self, df):
        if self.onehot:
            # 一种转换方式，对类别进行one-hot编码
            names = sorted(set(n for n in df.values))
            mapping = {name: i for i, name in enumerate(names)}
            x = torch.zeros(len(df), len(mapping))
            for i, n in enumerate(df.values):
                x[i, mapping[n]] = 1
            return x
        else:
            # 一种转换方式，将节点类别名称特征通过预训练的大模型进行表示
            x = self.model.encode(df.values, show_progress_bar=True,
                                  convert_to_tensor=True, device=self.device)
            return x.cpu()


def data_establish(defect_path, structure_path, segment_path):
    # 节点数据提取与处理
    defect_x, defect_y, defect_mapping = node_preprocess(defect_path, index_col="defectID",
                                                         encoders=[SimpleEncoder(index=["type", "information"]),
                                                                   IdentityEncoder(index="level", dtype=torch.long)])
    _, _, structure_mapping = node_preprocess(structure_path, index_col="lineID")
    _, _, segment_mapping = node_preprocess(structure_path, index_col="intervalID")

    # 关系数据提取与处理
    structure_segment, _ = edge_preprocess(structure_path, "lineID", structure_mapping,
                                           "intervalID", segment_mapping)
    segment_defect, _ = edge_preprocess(segment_path, "intervalID", segment_mapping,
                                        "defectID", defect_mapping)

    # 图数据集构建
    data = HeteroData()
    data["defect"].x, data["defect"].y = defect_x, defect_y
    data["line"].num_nodes, data["interval"].num_nodes = len(structure_mapping), len(segment_mapping)
    data["line"].node_id, data["interval"].node_id = torch.arange(len(structure_mapping)), torch.arange(len(segment_mapping))
    data["line", "contains", "interval"].edge_index = structure_segment
    data["interval", "has", "defect"].edge_index = segment_defect
    data.validate(raise_on_error=True)  # 图数据检查
    # 上述构建得到的关系连接方向是单向的，需确保双向使特征可以双向融合，这个双向是必要的
    # 也可添加额外的处理，可选
    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)
    # data = T.NormalizeFeatures()(data)  # 会对病害类型的onehot编码产生影响
    return data


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, embed_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), embed_channels)
        self.classifier = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        y = self.classifier(x)
        return x, y


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, embed_channels, out_channels, data, aggr):
        super().__init__()
        # 1. 对于结构特征，在模型训练过程中采用torch.nn.Embedding生成并学习
        # self.structure_emb = torch.nn.Embedding(data["line"].num_nodes, embed_channels)
        # self.segment_emb = torch.nn.Embedding(data["interval"].num_nodes, embed_channels)
        # 2. 手动设置初始值为0
        structure_feat = torch.zeros(data["line"].num_nodes, embed_channels)
        self.structure_emb = torch.nn.Embedding.from_pretrained(structure_feat, freeze=False)
        segment_feat = torch.zeros(data["interval"].num_nodes, embed_channels)
        self.segment_emb = torch.nn.Embedding.from_pretrained(segment_feat, freeze=False)

        self.gnn = GNN(hidden_channels, embed_channels, out_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata(), aggr=aggr)

    def forward(self, data):
        x_dict = {
            "line": self.structure_emb(data["line"].node_id),
            "interval": self.segment_emb(data["interval"].node_id),
            "defect": data["defect"].x
        }
        x_dict, y_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict, y_dict


def main(args):
    # 数据集构建与划分
    tunnel_data = data_establish(args.node1, args.rel1, args.rel2)
    # print(tunnel_data)
    node_transform = T.RandomNodeSplit(num_val=1, num_test=2)  # 仅处理包含标签也就是y的节点
    node_splits = node_transform(tunnel_data)
    # print(node_splits["defect"].train_mask, node_splits["defect"].val_mask, node_splits["defect"].test_mask, sep="\n")

    # 模型构建
    tunnel_model = Model(args.hidden, args.embed, args.cls, tunnel_data, args.aggr)

    # 模型训练初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # 报错了可以用cpu运行看具体的报错内容
    # print("Device:", device)
    tunnel_model = tunnel_model.to(device)
    optimizer = torch.optim.AdamW(tunnel_model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    res = None
    start_time = time.time()
    # 模型训练
    for epoch in range(args.epochs):
        tunnel_model.train()
        optimizer.zero_grad()
        tunnel_data = tunnel_data.to(device)
        _, output = tunnel_model(tunnel_data)
        train_mask = node_splits["defect"].train_mask
        loss = criterion(output["defect"][train_mask], tunnel_data["defect"].y[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tunnel_model.eval()
            _, preds = tunnel_model(tunnel_data)
            pred = preds["defect"].softmax(dim=-1).argmax(dim=-1)
            test_mask = node_splits["defect"].test_mask
            test_correct = pred[test_mask] == tunnel_data["defect"].y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(tunnel_model, "tunnel_kg_best.pt")
        if epoch == args.epochs - 1:
            # torch.save(tunnel_model, "tunnel_kg_last.pt")
            res = torch.cat((preds["interval"].softmax(dim=-1).argmax(dim=-1), preds["line"].softmax(dim=-1).argmax(dim=-1)))
        # print(f" Epoch: [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}, Accuracy: {test_acc:.4f} ")

    # 训练结束
    # print(f"total training time {(time.time() - start_time):.3f} s")
    return res.cpu().numpy().tolist()


@torch.no_grad()
def predict(args, visualize=False):
    """
    查看数据结构，可视化图，输出预测结果
    """
    data = data_establish(args.node1, args.rel1, args.rel2)
    # print(data.x_dict, data.edge_index_dict, sep="\n")
    if visualize:
        visualize_graph_nx(data)
        return

    model = torch.load("tunnel_kg_last.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data, model = data.to(device), model.to(device)
    _, out = model(data)
    print(out["defect"].softmax(dim=-1).argmax(dim=-1))
    print(out["interval"].softmax(dim=-1).argmax(dim=-1))
    print(out["line"].softmax(dim=-1).argmax(dim=-1))


def visualize_graph_nx(data):
    nx_graph = to_networkx(data)
    plt.figure(figsize=(12,12))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(nx_graph, pos=nx.spring_layout(nx_graph, seed=45), with_labels=False, node_color=data.y,
                     cmap="Set2")
    # plt.savefig("kg.png")
    plt.show()


parser = argparse.ArgumentParser("Heterogeneous GNN model training")
# data
parser.add_argument("--node1", default="image.csv", type=str, help="node data path")
parser.add_argument("--rel1", default="line_interval.csv", type=str, help="relation data path")
parser.add_argument("--rel2", default="interval_defect.csv", type=str)
# model
parser.add_argument("--hidden", default=8, type=int, help="model hidden channel")
parser.add_argument("--embed", default=8, type=int, help="embedding channel")
parser.add_argument("--cls", default=2, type=int, help="classification number")
parser.add_argument("--aggr", default="mean", type=str, help="message aggregation method for Heterogeneous GNN")
# train
parser.add_argument("--epochs", default=100, type=int, help="model training epochs")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")


if __name__ == '__main__':
    # simple_example()
    # simple_llm()


    # 由于数据量较少，模型每次训练会随机初始化参数，导致结果有波动
    # 但模型很小，训练速度很快，所以可以训练一定次数取最合理的
    ps = parser.parse_args()
    predict(ps, visualize=True)
    # stac = {}
    # for q in tqdm(range(50)):
    #     result = main(ps)
    #     result_str = ''.join(str(i) for i in result)
    #     if result_str not in stac:
    #         stac[result_str] = 1
    #     else:
    #         stac[result_str] += 1
    # stac = sorted(stac.items(), key=lambda x: x[1], reverse=True)
    # print(stac)


