import json
import logging
from collections import deque, defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DynamicDAGScheduler:
    def __init__(self, toolset):
        """
        初始化调度器
        :param toolset: 实例化后的 Toolset 对象 (包含 CGT, SCP, FCV 等方法)
        """
        self.toolset = toolset
        # 定义熔断阈值
        self.CRITICAL_SOURCE_THRESHOLD = 0.2

    def _validate_and_sort_dag(self, plan_json):
        """
        [核心逻辑] 解析 DAG 并进行拓扑排序，决定执行顺序。
        """
        try:
            tools = plan_json.get('tools', [])
            dependencies = plan_json.get('dependencies', [])

            if not tools:
                return True, []

            graph = defaultdict(list)
            in_degree = {t: 0 for t in tools}

            for upstream, downstream in dependencies:
                if upstream in tools and downstream in tools:
                    graph[upstream].append(downstream)
                    in_degree[downstream] += 1

            queue = deque([t for t in tools if in_degree[t] == 0])
            sorted_execution_order = []

            while queue:
                node = queue.popleft()
                sorted_execution_order.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if len(sorted_execution_order) != len(tools):
                logger.warning("Cycle detected or incomplete DAG. Fallback to linear execution.")
                return False, tools

            return True, sorted_execution_order

        except Exception as e:
            logger.error(f"DAG Validation Error: {e}")
            return False, plan_json.get('tools', [])

    def execute(self, plan_json, initial_text, retrieved_evidence):
        """
        执行整个流水线 (优化版：增加并行执行与实时反馈)
        """
        print(f"\n{'=' * 10} Starting New Inference Trace {'=' * 10}")

        # 1. 确定执行顺序
        is_valid, execution_queue = self._validate_and_sort_dag(plan_json)

        # 2. 初始化“黑板”
        blackboard = {
            "initial_text": initial_text,
            "retrieved_evidence": retrieved_evidence,
            "tool_outputs": {},
            "execution_log": [],
            "risk_flags": []
        }

        # 3. 执行工具 (并行化处理提高速度)
        # 注意：CPE 必须最后执行，因为它依赖其他工具的结果
        tools_to_run = [t for t in execution_queue if t != "CPE"]

        if tools_to_run:
            print(f"[*] Planning to run {len(tools_to_run)} tools: {tools_to_run}")
            print(f"[*] ThreadPool activated. Waiting for API responses...")

            # 使用线程池并发调用 API
            with ThreadPoolExecutor(max_workers=len(tools_to_run)) as executor:
                future_to_tool = {}
                for tool_name in tools_to_run:
                    tool_func = getattr(self.toolset, tool_name, None)
                    if tool_func:
                        # 提交到线程池
                        future = executor.submit(tool_func, blackboard)
                        future_to_tool[future] = tool_name
                    else:
                        print(f" [!] Tool {tool_name} not found in toolset.")

                for future in as_completed(future_to_tool):
                    t_name = future_to_tool[future]
                    try:
                        result = future.result()
                        blackboard["tool_outputs"][t_name] = result
                        # 实时打印完成状态
                        status_tag = "PASS" if result.get('credibility_score', 1.0) > 0.5 else "RISK"
                        print(f"   [+] Tool Completed: {t_name:<5} | Status: {status_tag}", flush=True)

                        # 记录日志
                        blackboard["execution_log"].append({
                            "tool": t_name, "status": "success", "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"   [x] Tool Failed: {t_name:<5} | Error: {str(e)[:50]}...", flush=True)
                        blackboard["execution_log"].append({"tool": t_name, "status": "failed", "error": str(e)})

        # 4. 确保 CPE (最终判决) 被执行 (CPE 依赖前序所有结果，必须串行)
        if "CPE" not in blackboard["tool_outputs"]:
            print(f"[*] Running Final Aggregator: CPE ...", end="", flush=True)
            try:
                final_verdict = self.toolset.CPE(blackboard)
                blackboard["tool_outputs"]["CPE"] = final_verdict
                print(" Done!")
            except Exception as e:
                print(f" Failed!")
                logger.error(f"Critical Error in CPE: {e}")

        print(f"{'=' * 35}\n")
        return blackboard