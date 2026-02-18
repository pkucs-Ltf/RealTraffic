#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主运行文件 - 用于任务选择和参数设置
整套代码的入口点，支持三种任务：
1. 仿真任务 (simulation)
2. 交通灯控优化任务 (traffic_light_optimization) 
3. 变道策略优化任务 (lane_change_optimization)
"""

import argparse
import sys
import os
import yaml
import logging
from typing import Dict, Any

# 导入任务类
from task.simu_task import SimuTask
from task.opti_task_TLC import OptiTaskTLC
from task.lane_change_task import LaneChangeTask  # noqa: F401 – used via available_tasks

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainRunner")


class TaskRunner:
    """任务运行器主类"""
    
    def __init__(self):
        """初始化任务运行器"""
        self.available_tasks = {
            'simulation': {
                'class': SimuTask,
                'config': 'configs\\simu_Manha.yml',
                'description': '仿真任务 - 城市区域的仿真工作流'
            },
            
            'traffic_light_optimization': {
                'class': OptiTaskTLC,
                'config': 'configs/opti_task_TLC_Manha_dqn.yml',
                'description': '交通灯控优化任务 - 单独交通优化任务'
            },
            'lane_change_optimization': {
                'class': LaneChangeTask,
                'config': 'configs/lane_change_task_DC.yml',
                'description': '变道策略优化任务 - 道路资源协同优化'
            },
        }
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_file (str): 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_file}: {e}")
            return {}
    
    def validate_task_type(self, task_type: str) -> bool:
        """
        验证任务类型
        
        Args:
            task_type (str): 任务类型
            
        Returns:
            bool: 是否有效
        """
        return task_type in self.available_tasks
    
    def list_available_tasks(self):
        """列出所有可用的任务"""
        print("\n可用的任务类型:")
        print("=" * 60)
        for task_type, info in self.available_tasks.items():
            print(f"任务类型: {task_type}")
            print(f"描述: {info['description']}")
            print(f"配置文件: {info['config']}")
            print("-" * 60)
    
    def run_task(self, task_type: str, config_file: str = None, 
                 config_override: Dict = None):
        """
        运行指定的任务
        
        Args:
            task_type (str): 任务类型
            config_file (str): 配置文件路径（可选）
            config_override (Dict): 配置覆盖（可选）
            
        Returns:
            Any: 任务执行结果
        """
        if not self.validate_task_type(task_type):
            logger.error(f"无效的任务类型: {task_type}")
            self.list_available_tasks()
            return None
        
        task_info = self.available_tasks[task_type]
        
        # 确定配置文件
        if config_file is None:
            config_file = task_info['config']
        
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            logger.error(f"配置文件不存在: {config_file}")
            return None
        
        logger.info(f"开始执行任务: {task_type}")
        logger.info(f"使用配置文件: {config_file}")
        
        try:
            # 加载配置
            config = self.load_config(config_file)
            
            # 应用配置覆盖
            if config_override:
                config.update(config_override)
                logger.info("应用了配置覆盖参数")
            
            # 创建并运行任务
            task_class = task_info['class']
            task_instance = task_class(config_dict=config)
            
            logger.info(f"正在运行 {task_info['description']}...")
            result = task_instance.run()
            
            logger.info(f"任务 {task_type} 执行完成")
            return result
            
        except Exception as e:
            logger.error(f"执行任务 {task_type} 时出错: {e}", exc_info=True)
            return None
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n=== 交互式任务运行器 ===")
        self.list_available_tasks()
        
        while True:
            try:
                task_type = input("\n请选择任务类型 (输入 'quit' 退出): ").strip()
                
                if task_type.lower() == 'quit':
                    print("退出程序")
                    break
                
                if not self.validate_task_type(task_type):
                    print(f"无效的任务类型: {task_type}")
                    continue
                
                # 询问是否使用自定义配置文件
                use_custom_config = input("是否使用自定义配置文件? (y/N): ").strip().lower()
                config_file = None
                
                if use_custom_config == 'y':
                    config_file = input("请输入配置文件路径: ").strip()
                    if not os.path.exists(config_file):
                        print(f"配置文件不存在: {config_file}")
                        continue
                
                # 运行任务
                result = self.run_task(task_type, config_file)
                
                if result:
                    print(f"\n任务执行结果:")
                    print("-" * 40)
                    for key, value in result.items():
                        if isinstance(value, (str, int, float)):
                            print(f"{key}: {value}")
                    print("-" * 40)
                
                # 询问是否继续
                continue_choice = input("\n是否继续运行其他任务? (Y/n): ").strip().lower()
                if continue_choice == 'n':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n用户中断，退出程序")
                break
            except Exception as e:
                logger.error(f"交互模式出错: {e}")


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="城市交通仿真与优化任务运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的任务类型:
  simulation                 - 仿真任务
  traffic_light_optimization - 交通灯控优化任务
  lane_change_optimization   - 变道策略优化任务

示例用法:
  python run.py -t simulation
  python run.py -t traffic_light_optimization -c configs/opti_task_TLC_Manha_dqn.yml
  python run.py -t lane_change_optimization -c configs/lane_change_task_DC.yml
  python run.py --interactive
  python run.py --list-tasks
        """
    )
    
    parser.add_argument(
        '-t', '--task-type',
        type=str,
        choices=['simulation', 'traffic_light_optimization', 'lane_change_optimization'],
        default='simulation',
        help='任务类型 (默认: simulation)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='配置文件路径（可选，默认使用对应任务的默认配置）'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='启动交互式模式'
    )
    
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='列出所有可用的任务类型'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建任务运行器
    runner = TaskRunner()
    
    # 处理命令行参数
    if args.list_tasks:
        runner.list_available_tasks()
        return
    
    if args.interactive:
        runner.interactive_mode()
        return
    
    
    if args.task_type:
        result = runner.run_task(args.task_type, args.config)
        if result is None:
            sys.exit(1)
        return
    
    # 如果没有指定任何参数，显示帮助信息
    parser.print_help()


if __name__ == "__main__":
    main()
