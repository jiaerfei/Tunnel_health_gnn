# Tunnel_health_gnn

## A data knowledge-driven shield tunnel structure health evaluation method by graph neural network

<img width="1158" height="967" alt="route" src="https://github.com/user-attachments/assets/234c7989-8905-4a53-8636-bc8adf2ec2d8" />


### 1. 环境配置(必需的)
- ultralytics (https://github.com/ultralytics/ultralytics)
- PyG (https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- py2neo (pip install py2neo)
> py2neo不再维护更新，Neo4j目前有官方推荐的管理库：pip install neo4j


### 2. 软件配置(必需的)
Neo4j (https://neo4j.com/docs/operations-manual/current/installation/windows/)


### 3. 文件说明
#### main_box.py
病害图像的自动识别、量化信息提取与存储

#### main_tunnel_kg.py
读取病害识别结果文件与评估知识，进而构建知识图谱

#### main_tunnel_gnn.py
GNN输入数据处理、网络搭建、训练与预测

#### 其他为数据、模型最优权重等文件


### 4. 数据集下载
待更新
