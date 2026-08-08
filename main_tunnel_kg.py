from py2neo import Graph
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd


class TunnelGraph:

    def __init__(self, loc="bolt://localhost:7687", user="neo4j", password="neo4jforlearn", database_name="neo4j",
                 defect_file="image.csv", maintenance_file="maintenance_strategy.csv"):
        """
        :param loc: 浏览器地址（见cmd终端）
        :param user: 浏览器用户名称（自定义）
        :param password: 浏览器密码（自定义）
        :param database_name: 浏览器界面左侧数据库名称（一般默认即可）
        :param defect_file: 病害识别结果文件路径
        :param maintenance_file: 维养策略文件路径
        :return: None
        """
        self.graph = Graph(profile=loc, auth=(user, password), name=database_name)
        self.defect_file = defect_file
        self.maintenance_file = maintenance_file


    def create_graph(self):
        # 读取病害识别结果，自动转换存储进知识图谱
        defect_csv = pd.read_csv(self.defect_file)
        self.graph.run("create (:Tunnel {Name: 'Shanghai'})")
        # 创建实体
        for i in tqdm(range(len(defect_csv))):
            cypher_node1 = "merge (:Line {{ID: {}, Year: 2026}})".format(defect_csv.iloc[i]["line"])
            cypher_node2 = "\n" + "merge (:Interval {{ID: {}}})".format(defect_csv.iloc[i]["interval"])
            cypher_node3 = "\n" + "create (:Defect {{ID: {}, Type: '{}', Loc: {}, Info: {}, Image: '{}'}})".format(
                defect_csv.iloc[i]["defectID"], defect_csv.iloc[i]["type"], defect_csv.iloc[i]["location"],
                defect_csv.iloc[i]["information"], defect_csv.iloc[i]["image"])
            cypher = cypher_node1 + cypher_node2 + cypher_node3
            self.graph.run(cypher)

        # 创建关系
        for i in tqdm(range(len(defect_csv))):
            cypher_match1 = "match (a:Line {{ID: {}}})".format(defect_csv.iloc[i]["line"])
            cypher_match2 = "\n" + "match (b:Interval {{ID: {}}})".format(defect_csv.iloc[i]["interval"])
            cypher_match3 = "\n" + "match (c:Defect {{ID: {}}})".format(defect_csv.iloc[i]["defectID"])
            cypher_match4 = "\n" + "match (d:Tunnel {Name: 'Shanghai'})"

            cypher_create1 = "\n" + "merge (a)-[:CONSIST_OF]->(b)"
            cypher_create2 = "\n" + "merge (b)-[:HAS_DEFECT]->(c)"
            cypher_create3 = "\n" + "merge (d)-[:HAS_LINE]->(a)"
            cypher = (cypher_match1 + cypher_match2 + cypher_match3 + cypher_match4
                      + cypher_create1 + cypher_create2 + cypher_create3)
            self.graph.run(cypher)


    def update_strategy(self):
        # 将维养策略转为图谱数据格式
        maintenance_csv = pd.read_csv(self.maintenance_file)
        self.graph.run("create (:Standard {Name: 'CJJ_T 289-2018', Year: 2026})")
        cypher_match = ("match (a:Standard {Year: 2026})" + "match (b:Tunnel {Name: 'Shanghai'})" +
                        "merge (b)-[:DEAL_WITH]->(a)")
        self.graph.run(cypher_match)

        # 创建维养策略实体
        for i in tqdm(range(len(maintenance_csv))):
            cypher_node1 = "create (:Strategy {{Name: '{}', Level: {}}})".format(maintenance_csv.iloc[i]["Strategy"],
                                                                                   maintenance_csv.iloc[i]["Level"])
            cypher_node2 = "\n" + "create (:Object {{Name: '{}', min: {}, max: {}}})".format(
                maintenance_csv.iloc[i]["Object1"],
                maintenance_csv.iloc[i]["Cmin"], maintenance_csv.iloc[i]["Cmax"])
            cypher_node3 = "\n" + "create (:Object {{Name: '{}', min: {}, max: {}}})".format(
                maintenance_csv.iloc[i]["Object2"],
                maintenance_csv.iloc[i]["Lmin"], maintenance_csv.iloc[i]["Lmax"])
            cypher_node4 = "\n" + "create (:Object {{Name: '{}', min: {}, max: {}}})".format(
                maintenance_csv.iloc[i]["Object3"],
                maintenance_csv.iloc[i]["Smin"], maintenance_csv.iloc[i]["Smax"])
            cypher_node5 = "\n" + "merge (:Method {{Name: '{}', Price: {}, Level: {}}})".format(
                maintenance_csv.iloc[i]["Method_C1"], maintenance_csv.iloc[i]["Price_C1"],
                maintenance_csv.iloc[i]["Level"])
            cypher_node6 = "\n" + "merge (:Method {{Name: '{}', Price: {}, Level: {}}})".format(
                maintenance_csv.iloc[i]["Method_C2"], maintenance_csv.iloc[i]["Price_C2"],
                maintenance_csv.iloc[i]["Level"])

            cypher = cypher_node1 + cypher_node2 + cypher_node3 + cypher_node4 + cypher_node5 + cypher_node6
            self.graph.run(cypher)

        # 创建隧道与各策略关系
        for i in tqdm(range(len(maintenance_csv))):
            cypher_match1 = "match (a:Standard {Year: 2026})"
            cypher_match2 = "\n" + "match (b:Strategy {{Level: {}}})".format(maintenance_csv.iloc[i]["Level"])
            cypher_match3 = "\n" + "match (c:Object {{Name: '{}'}})".format(maintenance_csv.iloc[i]["Object1"])
            cypher_match4 = "\n" + "match (d:Object {{Name: '{}'}})".format(maintenance_csv.iloc[i]["Object2"])
            cypher_match5 = "\n" + "match (e:Object {{Name: '{}'}})".format(maintenance_csv.iloc[i]["Object3"])
            cypher_match6 = "\n" + "match (f:Method {{Name: '{}'}})".format(maintenance_csv.iloc[i]["Method_C1"])
            cypher_match7 = "\n" + "match (g:Method {{Name: '{}'}})".format(maintenance_csv.iloc[i]["Method_C2"])

            cypher_create1 = "\n" + "merge (a)-[:HAS_LEVEL]->(b)"
            cypher_create2 = "\n" + "merge (b)-[:BASED_ON]->(c)"
            cypher_create3 = "\n" + "merge (b)-[:BASED_ON]->(d)"
            cypher_create4 = "\n" + "merge (b)-[:BASED_ON]->(e)"
            cypher_create5 = "\n" + "merge (a)-[:HAS_WAY]->(f)"
            cypher_create6 = "\n" + "merge (a)-[:HAS_WAY]->(g)"

            cypher = (cypher_match1 + cypher_match2 + cypher_match3 + cypher_match4 + cypher_match5 + cypher_match6 +
                      cypher_match7 + cypher_create1 + cypher_create2 + cypher_create3 + cypher_create4 +
                      cypher_create5 + cypher_create6)
            self.graph.run(cypher)


    def update_health(self):
        # TODO：图神经网络得到病害评估结果后将其补充到知识图谱属性中去，对于病害节点直接检索赋值即可
        pass


if __name__ == '__main__':
    under_graph = TunnelGraph()
    # 1. 创建知识图谱
    under_graph.create_graph()
    # 2. 更新维养策略
    under_graph.update_strategy()
    print(f"知识图谱节点数量为{len(NodeMatcher(graph=under_graph.graph))}个")
