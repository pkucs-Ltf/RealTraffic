import traci
import os
import argparse
import time
import pickle
import numpy as np

def load_pickle_file(filepath):
    """安全地加载pickle文件。"""
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件未找到: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class SimulationRunner:
    """
    一个简单的SUMO仿真运行器，用于计算交通指标和准确率。
    """
    def __init__(self, net_file, route_file, end_time=3600, use_gui=False, sumo_binary_path='sumo', real_data_file=None, edge_limit_file=None):
        """
        初始化仿真运行器。

        :param net_file: SUMO路网文件 (.net.xml)
        :param route_file: SUMO车流文件 (.rou.xml)
        :param end_time: 仿真结束时间 (秒)
        :param use_gui: 是否使用SUMO GUI
        :param sumo_binary_path: SUMO可执行文件路径 ('sumo' 或 'sumo-gui')
        :param real_data_file: 包含真实交通状况数据的pickle文件
        :param edge_limit_file: 包含道路速度限制的pickle文件
        """
        self.net_file = net_file
        self.route_file = route_file
        self.end_time = end_time
        self.use_gui = use_gui
        self.sumo_binary = f"{sumo_binary_path}-gui" if use_gui else sumo_binary_path
        
        self.inside_vehicles = {}
        self.completed_trips = {}
        self.conn = None
        
        self.real_data = load_pickle_file(real_data_file) if real_data_file else None
        self.edge_limit_data = load_pickle_file(edge_limit_file) if edge_limit_file else None

    def _start_connection(self):
        """启动SUMO并建立Traci连接。"""
        if not os.path.exists(self.net_file):
            raise FileNotFoundError(f"路网文件未找到: {self.net_file}")
        if not os.path.exists(self.route_file):
            raise FileNotFoundError(f"车流文件未找到: {self.route_file}")

        sumo_cmd = [self.sumo_binary, "-n", self.net_file, "-r", self.route_file, "--time-to-teleport", "-1", "--no-warnings", "true"]
        if self.use_gui:
            sumo_cmd.append("--start")
            
        traci.start(sumo_cmd)
        self.conn = traci
        print("Traci连接已成功建立。")

    def track_vehicle_movements(self):
        """每个仿真步骤调用，以跟踪车辆的进入和退出。"""
        current_time = self.conn.simulation.getTime()
        for vehicle_id in self.conn.simulation.getDepartedIDList():
            self.inside_vehicles[vehicle_id] = current_time
        for vehicle_id in self.conn.simulation.getArrivedIDList():
            if vehicle_id in self.inside_vehicles:
                self.completed_trips[vehicle_id] = current_time - self.inside_vehicles.pop(vehicle_id)

    def collect_metrics(self):
        """收集并计算当前时间步的各项交通指标。"""
        vehicle_ids = self.conn.vehicle.getIDList()
        mean_travel_time = sum(self.completed_trips.values()) / len(self.completed_trips) if self.completed_trips else 0
        total_waiting_time = sum(self.conn.vehicle.getWaitingTime(v_id) for v_id in vehicle_ids)
        total_queue_length = sum(self.conn.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.conn.lane.getIDList())
        speeds = [self.conn.vehicle.getSpeed(v_id) for v_id in vehicle_ids]
        mean_speed = sum(speeds) / len(speeds) if speeds else 0
        
        return {
            'time': self.conn.simulation.getTime(), 'vehicles_active': self.conn.vehicle.getIDCount(),
            'vehicles_completed': len(self.completed_trips), 'mean_travel_time': mean_travel_time,
            'total_waiting_time': total_waiting_time, 'total_queue_length': total_queue_length, 'mean_speed': mean_speed
        }

    def calculate_congestion_score(self, average_speeds, average_waiting_vehicles):
        """根据仿真结果和道路限速计算拥堵分数。"""
        if not self.edge_limit_data:
            print("警告: 未提供道路限速文件，无法计算拥堵分数。")
            return {}

        congestion_scores = {}
        for road_id, avg_speed in average_speeds.items():
            if road_id in self.edge_limit_data:
                limit = self.edge_limit_data[road_id]
                avg_waiting = average_waiting_vehicles.get(road_id, 0)
                
                if avg_speed == 0:
                    congestion_scores[road_id] = 0
                    continue

                ratio = limit / avg_speed
                if ratio > 2 or avg_waiting > 20:
                    congestion_scores[road_id] = 5
                elif 1.5 < ratio <= 2 or avg_waiting > 10:
                    congestion_scores[road_id] = 3
                else:
                    congestion_scores[road_id] = 0
        return congestion_scores

    def evaluate_accuracy(self, sumo_congestion_data):
        """将仿真拥堵数据与真实数据进行比较，计算F1分数、精确率和召回率。"""
        if not self.real_data:
            print("警告: 未提供真实数据文件，无法进行评估。")
            return None

        real_values, sumo_values = [], []
        for road_id, real_value in self.real_data.items():
            if road_id in sumo_congestion_data:
                real_values.append(real_value)
                sumo_values.append(sumo_congestion_data[road_id])

        if not real_values:
            print("警告: 仿真结果中没有与真实数据匹配的路段。")
            return None
        
        real_values, sumo_values = np.array(real_values), np.array(sumo_values)
        tp = np.sum((sumo_values != 0) & (real_values != 0))
        fp = np.sum((sumo_values != 0) & (real_values == 0))
        fn = np.sum((sumo_values == 0) & (real_values != 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n--- 仿真准确率评估 ---")
        print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return {'f1_score': f1, 'precision': precision, 'recall': recall}

    def run(self):
        """运行完整的仿真流程。"""
        self._start_connection()
        step, metrics_history, start_sim_time = 0, [], time.time()
        
        road_ids = [r_id for r_id in self.conn.edge.getIDList() if not r_id.startswith(':')]
        road_speeds = {r_id: [] for r_id in road_ids}
        road_waiting = {r_id: [] for r_id in road_ids}

        try:
            while self.conn.simulation.getMinExpectedNumber() > 0 and step < self.end_time:
                self.conn.simulation.step()
                self.track_vehicle_movements()

                for r_id in road_ids:
                    road_speeds[r_id].append(self.conn.edge.getLastStepMeanSpeed(r_id))
                    road_waiting[r_id].append(self.conn.edge.getLastStepHaltingNumber(r_id))
                
                if step % 100 == 0:
                    metrics = self.collect_metrics()
                    metrics_history.append(metrics)
                    print(f"时间: {metrics['time']}s | 活动车辆: {metrics['vehicles_active']} | "
                          f"平均旅行时间: {metrics['mean_travel_time']:.2f}s")
                step += 1
        except traci.TraCIException as e:
            print(f"仿真过程中发生错误: {e}")
        finally:
            self.close()
            print(f"\n仿真结束。总用时: {time.time() - start_sim_time:.2f}秒")
        
        avg_speeds = {r_id: np.mean(s) for r_id, s in road_speeds.items() if s}
        avg_waiting = {r_id: np.mean(w) for r_id, w in road_waiting.items() if w}
        return metrics_history, avg_speeds, avg_waiting

    def close(self):
        """关闭Traci连接。"""
        if self.conn:
            traci.close()
            self.conn = None
            print("Traci连接已关闭。")

def main():
    parser = argparse.ArgumentParser(description="运行SUMO仿真并计算交通指标及准确率。")
    parser.add_argument("-n", "--net-file", required=True, help="SUMO路网文件 (.net.xml)")
    parser.add_argument("-r", "--route-file", required=True, help="SUMO车流文件 (.rou.xml)")
    parser.add_argument("-e", "--end-time", type=int, default=3600, help="仿真结束时间 (秒)")
    parser.add_argument("--gui", action="store_true", help="如果指定，则使用SUMO GUI")
    parser.add_argument("--real-data-file", help="真实交通状况数据的pickle文件")
    parser.add_argument("--edge-limit-file", help="道路速度限制的pickle文件")
    
    args = parser.parse_args()
    
    runner = SimulationRunner(
        net_file=args.net_file, route_file=args.route_file, end_time=args.end_time,
        use_gui=args.gui, real_data_file=args.real_data_file, edge_limit_file=args.edge_limit_file
    )
    
    metrics, avg_speeds, avg_waiting = runner.run()
    
    if metrics:
        print("\n--- 仿真最终指标 ---")
        for key, value in metrics[-1].items():
            print(f"{key.replace('_', ' ').title()}: {value if isinstance(value, int) else f'{value:.2f}'}")

    if avg_speeds and avg_waiting:
        congestion = runner.calculate_congestion_score(avg_speeds, avg_waiting)
        if congestion:
            runner.evaluate_accuracy(congestion)

if __name__ == "__main__":
    main()
