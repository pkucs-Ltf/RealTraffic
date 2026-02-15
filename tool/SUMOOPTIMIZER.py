import logging
import os
import subprocess
import xml.etree.ElementTree as ET
import json
from datetime import datetime

from langchain_deepseek import ChatDeepSeek
os.environ['DEEPSEEK_API_KEY'] = 'sk-04f82ec63c5f4bfbbb72000e23b27faa'
# 设置 deepseek API key



class SUMOSignalOptimizer:
    def __init__(self,logger=None):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise Exception("请设置 DEEPSEEK_API_KEY")
        self.logger = logger or logging.getLogger(__name__)

    def get_netconvert_files(self, net_file):
        # 如果 net_file 是形如 "map.net.xml"，则去掉后缀后缀为 base
        if net_file.endswith(".net.xml"):
            base_name = net_file[:-8]  # 去除 ".net.xml"
        else:
            base_name, _ = os.path.splitext(net_file)
        return {
            "net": net_file,
            "node": f"{base_name}.nod.xml",
            "edge": f"{base_name}.edg.xml",
            "connection": f"{base_name}.con.xml",
            "tll": f"{base_name}.tll.xml"
        }

    def extract_intersection_info(self, intersection_id, net_file):
        """
        提取指定交叉口（信号灯）的相关信息，返回 JSON 字符串。
        """
        files = self.get_netconvert_files(net_file)
        result = {}
        # 1. 解析节点信息
        try:
            tree_nod = ET.parse(files["node"])
            root_nod = tree_nod.getroot()
        except Exception as e:
            raise Exception(f"解析节点文件 {files['node']} 出错: {e}")

        node_info = None
        for node in root_nod.findall('node'):
            if node.get('id') == intersection_id:
                node_info = node.attrib
                break
        result['node'] = node_info if node_info is not None else {}

        # 2. 解析边信息（与该交叉口相连）
        try:
            tree_edg = ET.parse(files["edge"])
            root_edg = tree_edg.getroot()
        except Exception as e:
            raise Exception(f"解析边文件 {files['edge']} 出错: {e}")

        edges_info = []
        for edge in root_edg.findall('edge'):
            if edge.get('from') == intersection_id or edge.get('to') == intersection_id:
                edges_info.append(edge.attrib)
        result['edges'] = edges_info

        # 获取所有与该路口相连的边 ID
        incident_edge_ids = {edge.get('id') for edge in edges_info}

        # 3. 解析连接信息
        try:
            tree_con = ET.parse(files["connection"])
            root_con = tree_con.getroot()
        except Exception as e:
            raise Exception(f"解析连接文件 {files['connection']} 出错: {e}")

        connections_info = []
        for con in root_con.findall('connection'):
            if con.get('from') in incident_edge_ids or con.get('to') in incident_edge_ids:
                connections_info.append(con.attrib)
        result['connections'] = connections_info

        # 4. 解析信号灯信息
        try:
            tree_tll = ET.parse(files["tll"])
            root_tll = tree_tll.getroot()
        except Exception as e:
            raise Exception(f"解析信号灯文件 {files['tll']} 出错: {e}")

        traffic_light_info = {}
        tl_logic = root_tll.find(f"tlLogic[@id='{intersection_id}']")
        if tl_logic is not None:
            tl_data = tl_logic.attrib.copy()
            phases = [phase.attrib for phase in tl_logic.findall('phase')]
            tl_data['phases'] = phases
            traffic_light_info['tlLogic'] = tl_data
        else:
            traffic_light_info['tlLogic'] = {}

        tl_connections = [con.attrib for con in root_tll.findall('connection') if con.get('tl') == intersection_id]
        traffic_light_info['connections'] = tl_connections

        result['traffic_light'] = traffic_light_info

        return json.dumps(result, indent=4, ensure_ascii=False)

    def assemble_prompt(self, intersection_json, preferred_phase, lane_observations, intersection_id):
        """
        组装任务 Prompt，将交叉口 JSON 数据、偏好相位、车道观测数据整合到 prompt 中，
        指示智能体输出分析和决策（符合 SUMO .con 文件格式的 connection 标签）。
        """
        prompt = f"""SUMO 交通信号灯分析与优化任务 Prompt

背景与任务概述：
你是一个专注于 SUMO（城市交通仿真）中潮汐车道配置优化的 AI 助手。
现有交叉口ID为 "{intersection_id}" 的数据如下，请根据任务要求进行综合分析，针对此路口输出潮汐车道优化建议。

输入数据格式：
1. 交叉口数据（JSON格式）：
{intersection_json}

2. 当前路口交通信号灯智能体决策时的偏好相位：
{preferred_phase}

3. 当前路口各车道平均观测数据：
{json.dumps(lane_observations, indent=4, ensure_ascii=False)}

任务要求：
-1：首先理解正在处理的路口的模式：读取交叉口数据，通过现有的connection标签，找出所有的进入边（from),和离开边(to),分析清楚各个边上的车道数量。
-2：接着读取当前路口交通信号灯智能体决策时的偏好相位：相位是一串字符串，其中每个状态字母都按顺序对应一个路口中实际存在的connection,序号从0开始。
- 你需要结合交叉口数据中的记录，计算偏好相位中绿灯（G）所在位置，注意从0开始，列一张表一一对应相位和路口中的connection，不可跳过，严格对应序号，避免出错，从而将每个相位状态映射到具体连接。
-映射完成后，偏好相位中G对应的连接就是高压连接。罗列出这些高压连接，分析清楚它们的进入边（from)和进入车道，以及离开边（to)及离开车道。
-3：在准确找出当前路口的所有高压连接并理解它们的模式后，进行具体的潮汐车道决策，方法如下：
-根据所有高压连接的进入边（from)及进入边上的车道，读取“当前路口各车道平均观测数据”，从而准确找出一条进入边车道的交通压力最大的连接作为决策连接。
-针对决策连接，再次确认它的进入边（from)及进入车道，离开边(to)及离开边上的所有车道。
-你的决策方法为：从决策连接的进入边（from)上再选取一条交通压力最低的车道作为出发车道，向离开边(to)上的所有车道新建连接。特别注意，必须是所有车道，这意味着每次从进入边（from)开始的标签不会只有一条，而是多条，顺次连接离开边（to)的所有车道。这些新连接的方向与决策连接一致，但使用了空余道路资源，所以可以减轻决策连接的压力。
-决策的格式为一系列标签：<connection from="[进入边]" to="[离开边]" fromLane="[进入车道]" toLane="[离开车道]"/>...   数量和离开边(to)的车道数一致，这些连接的出发车道都一样，但离开车道依次覆盖离开边的所有车道。
-4：特别注意，只有路口进入from边才能是出发边，所有的决策都应当在from边上的车道开始.to边和to边的车道不可互相连接。新增的连接的进入边和离开边必须和决策连接的进入边和离开边完全一致，变化的只是车道的选择。
-完成的决策应当避免和已经存在的连接重复，即依次检查出发边和目标边的已有链接，确保不重复。因为这不会带来任何改变，完成的决策必须在进入边和离开边已有的连接中验证不重复性。
-每次做出的决策，需要确保能够让出发边的低压连接和离开边的所有车道都有连接。
-一旦完成决策需要立刻反思是否遵循了上述约束：高压连接的序号计算是正确的；的确选取了入边车道压力最高的连接作为决策连接；新增的连接的出入边和决策连接一致，而且的确优先使用了同边上的其他低压车道；新增的连接的确尚不存在；做出的决策的确覆盖了离开边的所有车道。
-反思的过程需要在答案中输出。
- 5：请严格按照以下格式输出答案：
-----------------------------------------------------
分析：
[在此描述你的详细分析过程和发现]
-----------------------------------------------------
决策：
<connection from="[进入边]" to="[离开边]" fromLane="[进入车道A]" toLane="[离开车道B]"/>
<connection from="[进入边]" to="[离开边]" fromLane="[进入车道A]" toLane="[离开车道C]"/>
<connection from="[进入边]" to="[离开边]" fromLane="[进入车道A]" toLane="[离开车道D]"/>
...（决策部分只输出符合 SUMO .con 文件格式的新增 connection 标签，离开边上有几条车道就输出几条）
-----------------------------------------------------
请确保格式清晰，并标明对应的交叉口ID。
"""
        return prompt

    def run_llm(self, prompt):
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            max_tokens=1500
        )
        response = llm(prompt)
        return response

    def parse_llm_response(self, response):
        content = response.content if hasattr(response, "content") else str(response)
        parts = content.split("-----------------------------------------------------")
        analysis = ""
        decision = ""
        for part in parts:
            if "分析：" in part:
                analysis = part.split("分析：", 1)[1].strip()
            elif "决策：" in part:
                decision = part.split("决策：", 1)[1].strip()
        return {"analysis": analysis, "decision": decision}

    def append_decision_to_tree(self, decision_item, root):
        """
        将单个交叉口的分析与决策结果追加到 XML 树中，
        以注释的形式标明组头、分析、决策和组尾。
        """
        intersection_id = decision_item["intersection_id"]
        analysis = decision_item["analysis"]
        decision_xml = decision_item["decision"]

        header_comment = ET.Comment(f"=== Begin Analysis & Decision for Intersection {intersection_id} ===")
        root.append(header_comment)

        analysis_comment = ET.Comment(f"Analysis: {analysis}")
        root.append(analysis_comment)

        decision_header_comment = ET.Comment(f"Decision for Intersection {intersection_id}:")
        root.append(decision_header_comment)

        for line in decision_xml.splitlines():
            line = line.strip()
            if line.startswith("<connection") and line.endswith("/>"):
                try:
                    conn_elem = ET.fromstring(line)
                    root.append(conn_elem)
                except Exception as e:
                    print(f"解析连接标签失败: {line} 错误: {e}")
                    continue

        footer_comment = ET.Comment(f"=== End Analysis & Decision for Intersection {intersection_id} ===")
        root.append(footer_comment)

    def merge_network_files(self, net_file):
        """
        调用 netconvert 将各基本文件合成为一个新路网文件，新文件名在原文件名基础上添加版本号（如 v2、v3……）。
        """
        net_dir = os.path.dirname(net_file)
        basename = os.path.basename(net_file)
        if basename.endswith(".net.xml"):
            base_prefix = basename[:-8]
            ext = ".net.xml"
        else:
            base_prefix, ext = os.path.splitext(basename)
        version = 2
        new_net_file = os.path.join(net_dir, f"{base_prefix}v{version}{ext}")
        while os.path.exists(new_net_file):
            version += 1
            new_net_file = os.path.join(net_dir, f"{base_prefix}v{version}{ext}")

        files = self.get_netconvert_files(net_file)
        merge_cmd = f"netconvert --node-files={files['node']} --edge-files={files['edge']} --connection-files={files['connection']} --tls.layout incoming --output-file={new_net_file}.net.xml "
        print(f"执行合成命令: {merge_cmd}")
        ret = subprocess.run(merge_cmd, shell=True)
        if ret.returncode != 0:
            raise Exception(f"netconvert 合成命令执行失败: {merge_cmd}")
        return new_net_file

    def process(self, net_file_path, decisions_json):
        """
        主流程方法：
        1. 将 net_file_path 转换为绝对路径，保证无论在哪个目录运行都能找到。
        2. 在原始文件所在目录下创建一个 optimized 目录，并在其中为每次迭代创建一个专用子目录，避免覆盖。
        3. 调用 netconvert 拆分路网、使用决策更新连接，再合成新的路网文件。
        4. 返回新生成的网络文件路径。
        """
        # 如果 decisions_json 是字符串，则解析为字典
        if isinstance(decisions_json, str):
            decisions_json = json.loads(decisions_json)

        net_file_path = os.path.abspath(net_file_path)
        net_dir = os.path.dirname(net_file_path)
        basename = os.path.basename(net_file_path)
        if basename.endswith(".net.xml"):
            base_prefix = basename[:-8]
        else:
            base_prefix, _ = os.path.splitext(basename)

        # 创建 optimized 目录
        optimized_dir = os.path.join(net_dir, "optimized")
        os.makedirs(optimized_dir, exist_ok=True)

        # 根据当前时间创建一个专用的迭代目录，避免不同迭代结果覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iter_dir = os.path.join(optimized_dir, f"{base_prefix}_iter_{timestamp}")
        os.makedirs(iter_dir, exist_ok=True)

        # 设置拆分输出前缀为迭代目录下的 base_prefix
        split_prefix = os.path.join(iter_dir, base_prefix)

        # 1. 拆分路网
        split_cmd = f"netconvert --sumo-net-file {net_file_path} --plain-output-prefix {split_prefix}"
        print(f"执行拆分命令: {split_cmd}")
        # 注意：使用 net_dir 作为工作目录，确保相对路径正确
        ret = subprocess.run(split_cmd, shell=True, cwd=net_dir)
        if ret.returncode != 0:
            raise Exception(f"netconvert 拆分命令执行失败: {split_cmd}")

        # 生成基础文件的路径（拆分后的文件都在 iter_dir 中）
        files = self.get_netconvert_files(split_prefix)
        con_file = files["connection"]

        try:
            tree = ET.parse(con_file)
            root = tree.getroot()
        except Exception as e:
            raise Exception(f"加载连接文件 {con_file} 出错: {e}")

        # 2. 遍历 decisions_json 中的每个交叉口进行决策
        for intersection_id, data in decisions_json.items():
            preferred_phase = data.get("preferred_phase", "")
            lane_observations = data.get("lane_observations", {})
            try:
                intersection_json = self.extract_intersection_info(intersection_id, split_prefix)
                print(f"\n交叉口 {intersection_id} 解析得到的数据：")
                print(intersection_json)
            except Exception as e:
                print(f"交叉口 {intersection_id} 解析错误: {e}")
                continue

            prompt = self.assemble_prompt(intersection_json, preferred_phase, lane_observations, intersection_id)
            print("\n组装后的 Prompt：")
            print(prompt)

            print("\n调用 LLM 中，请稍候...")
            llm_response = self.run_llm(prompt)
            print("\nLLM 返回的原始结果：")
            print(llm_response.content if hasattr(llm_response, "content") else llm_response)



            parsed = self.parse_llm_response(llm_response)

            self.logger.info("LLM决策记录 - Intersection %s 分析: %s", intersection_id, parsed["analysis"])
            self.logger.info("LLM决策记录 - Intersection %s 决策: %s", intersection_id, parsed["decision"])

            print("\n解析后的回答：")
            print("分析部分：")
            print(parsed["analysis"])
            print("决策部分：")
            print(parsed["decision"])

            decision_item = {
                "intersection_id": intersection_id,
                "analysis": parsed["analysis"],
                "decision": parsed["decision"]
            }
            self.append_decision_to_tree(decision_item, root)

        # 更新连接文件
        tree.write(con_file, encoding="utf-8", xml_declaration=True)
        print(f"已更新连接文件: {con_file}")

        # 3. 合成新路网文件，生成文件名中带有迭代标识（时间戳）
        new_net_file = os.path.join(iter_dir, f"{base_prefix}_v{timestamp}.net.xml")
        merge_cmd = (
            f"netconvert --node-files {split_prefix}.nod.xml "
            f"--edge-files {split_prefix}.edg.xml "
            f"--connection-files {split_prefix}.con.xml "
            f"--output-file {new_net_file} "
            #f"--tls.layout incoming"
        )
        print(f"执行合并命令: {merge_cmd}")
        ret = subprocess.run(merge_cmd, shell=True, cwd=net_dir)
        if ret.returncode != 0:
            raise Exception(f"netconvert 合并命令执行失败: {merge_cmd}")

        print(f"新生成的路网文件: {new_net_file}")
        print("所有处理完成。")
        return new_net_file


