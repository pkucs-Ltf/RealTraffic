"""
MCP服务器编排器
提供REST API接口来管理交通仿真优化任务
"""
import os
import sys
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path
import zipfile
import shutil

# FastAPI相关导入

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# 数据库/缓存相关导入
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("警告: Redis不可用，将使用内存存储")

# 添加父目录到路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.langgraph_flow import OptimizationAgent


# Pydantic模型定义
class OptimizationRequest(BaseModel):
    """优化请求模型"""
    net_file: str
    rou_file: str
    real_data_file: str
    od_file: Optional[str] = None
    real_data_type: str = "shp"
    max_iterations: int = 30
    config_override: Optional[Dict[str, Any]] = None


class OptimizationStatus(BaseModel):
    """优化状态模型"""
    job_id: str
    status: str  # pending, running, completed, failed
    iteration: int = 0
    current_score: float = float('inf')
    best_score: float = float('inf')
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    last_action: Optional[Dict[str, Any]] = None


class JobResult(BaseModel):
    """任务结果模型"""
    job_id: str
    success: bool
    iterations: int
    best_score: float
    best_params: Dict[str, Any]
    history: List[Dict[str, Any]]
    final_files: Dict[str, str]
    artifacts_path: Optional[str] = None


class OptimizationOrchestrator:
    """优化任务编排器"""
    
    def __init__(self, config_file: str = "configs/langgraph.yaml"):
        """
        初始化编排器
        
        Args:
            config_file: 配置文件路径
        """
        self.app = FastAPI(
            title="SUMO优化智能体API",
            description="基于LangGraph的交通仿真优化服务",
            version="1.0.0"
        )
        
        # 初始化存储
        self.storage = self._init_storage()
        
        # 任务存储
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # 工作目录
        self.work_dir = Path("workspace")
        self.work_dir.mkdir(exist_ok=True)
        
        # 设置路由
        self._setup_routes()
        
        # 设置日志
        self._setup_logging()
        
        self.logger.info("优化编排器初始化完成")
    
    def _init_storage(self):
        """初始化存储后端"""
        if REDIS_AVAILABLE:
            try:
                storage = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                storage.ping()  # 测试连接
                self.logger.info("使用Redis作为存储后端")
                return storage
            except Exception as e:
                self.logger.warning(f"Redis连接失败，使用内存存储: {e}")
        
        # 使用内存存储
        return {}
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/optimize", response_model=Dict[str, str])
        async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
            """启动优化任务"""
            try:
                # 生成任务ID
                job_id = str(uuid.uuid4())
                
                # 验证输入文件
                if not all(os.path.exists(f) for f in [request.net_file, request.rou_file, request.real_data_file]):
                    raise HTTPException(status_code=400, detail="一个或多个输入文件不存在")
                
                # 创建任务状态
                job_status = OptimizationStatus(
                    job_id=job_id,
                    status="pending",
                    start_time=datetime.now()
                )
                
                # 保存任务状态
                self._save_job_status(job_id, job_status)
                
                # 启动后台任务
                background_tasks.add_task(self._run_optimization_task, job_id, request)
                
                self.logger.info(f"优化任务{job_id}已启动")
                return {"job_id": job_id, "status": "pending"}
                
            except Exception as e:
                self.logger.error(f"启动优化任务失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status/{job_id}", response_model=OptimizationStatus)
        async def get_optimization_status(job_id: str):
            """获取优化任务状态"""
            try:
                status = self._get_job_status(job_id)
                if status is None:
                    raise HTTPException(status_code=404, detail="任务不存在")
                return status
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"获取任务状态失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/result/{job_id}", response_model=JobResult)
        async def get_optimization_result(job_id: str):
            """获取优化任务结果"""
            try:
                result = self._get_job_result(job_id)
                if result is None:
                    raise HTTPException(status_code=404, detail="任务结果不存在")
                return result
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"获取任务结果失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/artifact/{job_id}")
        async def download_artifacts(job_id: str):
            """下载任务产出文件"""
            try:
                artifact_path = self._create_artifact_package(job_id)
                if not artifact_path or not os.path.exists(artifact_path):
                    raise HTTPException(status_code=404, detail="产出文件不存在")
                
                return FileResponse(
                    artifact_path,
                    media_type='application/zip',
                    filename=f"optimization_result_{job_id}.zip"
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"下载产出文件失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/job/{job_id}")
        async def cancel_optimization(job_id: str):
            """取消优化任务"""
            try:
                success = self._cancel_job(job_id)
                if not success:
                    raise HTTPException(status_code=404, detail="任务不存在或无法取消")
                
                return {"message": f"任务{job_id}已取消"}
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"取消任务失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/jobs")
        async def list_jobs():
            """列出所有任务"""
            try:
                jobs = self._list_all_jobs()
                return {"jobs": jobs}
            except Exception as e:
                self.logger.error(f"列出任务失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    async def _run_optimization_task(self, job_id: str, request: OptimizationRequest):
        """运行优化任务（后台任务）"""
        try:
            # 更新状态为运行中
            status = self._get_job_status(job_id)
            status.status = "running"
            self._save_job_status(job_id, status)
            
            # 创建工作目录
            job_work_dir = self.work_dir / job_id
            job_work_dir.mkdir(exist_ok=True)
            
            # 复制输入文件到工作目录
            input_files = self._prepare_input_files(job_id, request)
            
            # 创建优化代理
            agent = OptimizationAgent()
            
            # 设置进度回调
            def progress_callback(iteration: int, score: float, action: Dict[str, Any]):
                status = self._get_job_status(job_id)
                status.iteration = iteration
                status.current_score = score
                status.progress = min(100.0, (iteration / request.max_iterations) * 100)
                status.last_action = action
                self._save_job_status(job_id, status)
            
            # 运行优化
            result = agent.optimize(
                net_file=input_files['net_file'],
                rou_file=input_files['rou_file'],
                real_data_file=input_files['real_data_file'],
                od_file=input_files.get('od_file'),
                real_data_type=request.real_data_type,
                max_iterations=request.max_iterations
            )
            
            # 更新最终状态
            status = self._get_job_status(job_id)
            if result['success']:
                status.status = "completed"
                status.best_score = result['best_score']
                status.progress = 100.0
                
                # 保存结果
                job_result = JobResult(
                    job_id=job_id,
                    success=True,
                    iterations=result['iterations'],
                    best_score=result['best_score'],
                    best_params=result['best_params'],
                    history=result['history'],
                    final_files=result['final_files']
                )
                self._save_job_result(job_id, job_result)
                
            else:
                status.status = "failed"
                status.error_message = result.get('error', '未知错误')
            
            status.end_time = datetime.now()
            self._save_job_status(job_id, status)
            
            self.logger.info(f"优化任务{job_id}完成，状态: {status.status}")
            
        except Exception as e:
            # 更新错误状态
            status = self._get_job_status(job_id)
            status.status = "failed"
            status.error_message = str(e)
            status.end_time = datetime.now()
            self._save_job_status(job_id, status)
            
            self.logger.error(f"优化任务{job_id}执行失败: {e}")
    
    def _prepare_input_files(self, job_id: str, request: OptimizationRequest) -> Dict[str, str]:
        """准备输入文件"""
        job_work_dir = self.work_dir / job_id
        input_dir = job_work_dir / "input"
        input_dir.mkdir(exist_ok=True)
        
        # 复制文件
        input_files = {}
        
        # 复制网络文件
        net_src = Path(request.net_file)
        net_dst = input_dir / net_src.name
        shutil.copy2(net_src, net_dst)
        input_files['net_file'] = str(net_dst)
        
        # 复制路由文件
        rou_src = Path(request.rou_file)
        rou_dst = input_dir / rou_src.name
        shutil.copy2(rou_src, rou_dst)
        input_files['rou_file'] = str(rou_dst)
        
        # 复制真实数据文件
        real_src = Path(request.real_data_file)
        real_dst = input_dir / real_src.name
        shutil.copy2(real_src, real_dst)
        input_files['real_data_file'] = str(real_dst)
        
        # 复制OD文件（如果有）
        if request.od_file:
            od_src = Path(request.od_file)
            od_dst = input_dir / od_src.name
            shutil.copy2(od_src, od_dst)
            input_files['od_file'] = str(od_dst)
        
        return input_files
    
    def _create_artifact_package(self, job_id: str) -> Optional[str]:
        """创建产出文件压缩包"""
        try:
            job_work_dir = self.work_dir / job_id
            if not job_work_dir.exists():
                return None
            
            # 创建压缩包
            zip_path = job_work_dir / f"optimization_result_{job_id}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加结果文件
                result = self._get_job_result(job_id)
                if result and result.final_files:
                    for file_type, file_path in result.final_files.items():
                        if os.path.exists(file_path):
                            zipf.write(file_path, f"{file_type}_{Path(file_path).name}")
                
                # 添加日志文件
                log_files = list(job_work_dir.glob("*.log"))
                for log_file in log_files:
                    zipf.write(log_file, f"logs/{log_file.name}")
                
                # 添加结果JSON
                result_json = job_work_dir / "result.json"
                if result:
                    with open(result_json, 'w', encoding='utf-8') as f:
                        json.dump(result.dict(), f, ensure_ascii=False, indent=2, default=str)
                    zipf.write(result_json, "result.json")
            
            return str(zip_path)
            
        except Exception as e:
            self.logger.error(f"创建产出包失败: {e}")
            return None
    
    def _save_job_status(self, job_id: str, status: OptimizationStatus):
        """保存任务状态"""
        status_data = status.dict()
        
        if isinstance(self.storage, dict):
            # 内存存储
            self.storage[f"status:{job_id}"] = json.dumps(status_data, default=str)
        else:
            # Redis存储
            self.storage.set(f"status:{job_id}", json.dumps(status_data, default=str), ex=86400)  # 24小时过期
    
    def _get_job_status(self, job_id: str) -> Optional[OptimizationStatus]:
        """获取任务状态"""
        try:
            if isinstance(self.storage, dict):
                # 内存存储
                status_json = self.storage.get(f"status:{job_id}")
            else:
                # Redis存储
                status_json = self.storage.get(f"status:{job_id}")
            
            if status_json:
                status_data = json.loads(status_json)
                return OptimizationStatus(**status_data)
            return None
        except Exception as e:
            self.logger.error(f"获取任务状态失败: {e}")
            return None
    
    def _save_job_result(self, job_id: str, result: JobResult):
        """保存任务结果"""
        result_data = result.dict()
        
        if isinstance(self.storage, dict):
            # 内存存储
            self.storage[f"result:{job_id}"] = json.dumps(result_data, default=str)
        else:
            # Redis存储
            self.storage.set(f"result:{job_id}", json.dumps(result_data, default=str), ex=86400*7)  # 7天过期
    
    def _get_job_result(self, job_id: str) -> Optional[JobResult]:
        """获取任务结果"""
        try:
            if isinstance(self.storage, dict):
                # 内存存储
                result_json = self.storage.get(f"result:{job_id}")
            else:
                # Redis存储
                result_json = self.storage.get(f"result:{job_id}")
            
            if result_json:
                result_data = json.loads(result_json)
                return JobResult(**result_data)
            return None
        except Exception as e:
            self.logger.error(f"获取任务结果失败: {e}")
            return None
    
    def _cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        try:
            status = self._get_job_status(job_id)
            if status and status.status in ["pending", "running"]:
                status.status = "cancelled"
                status.end_time = datetime.now()
                self._save_job_status(job_id, status)
                return True
            return False
        except Exception as e:
            self.logger.error(f"取消任务失败: {e}")
            return False
    
    def _list_all_jobs(self) -> List[Dict[str, Any]]:
        """列出所有任务"""
        try:
            jobs = []
            
            if isinstance(self.storage, dict):
                # 内存存储
                for key in self.storage.keys():
                    if key.startswith("status:"):
                        job_id = key.replace("status:", "")
                        status = self._get_job_status(job_id)
                        if status:
                            jobs.append(status.dict())
            else:
                # Redis存储
                for key in self.storage.scan_iter(match="status:*"):
                    job_id = key.replace("status:", "")
                    status = self._get_job_status(job_id)
                    if status:
                        jobs.append(status.dict())
            
            # 按开始时间排序
            jobs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
            return jobs
            
        except Exception as e:
            self.logger.error(f"列出任务失败: {e}")
            return []
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """启动服务器"""
        self.logger.info(f"启动优化服务器 {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# 用于直接运行服务器的入口点
if __name__ == "__main__":
    orchestrator = OptimizationOrchestrator()
    orchestrator.run_server() 