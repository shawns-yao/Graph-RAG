"""医疗领域Graph RAG评测运行器模板

此模板展示如何运行评测并分析结果。
需要根据实际项目的API接口进行调整。
"""

import json
from pathlib import Path
from typing import Any

# 假设的项目导入（需要根据实际项目调整）
# from agentic_graph_rag.service import PipelineService
# from neo4j import GraphDatabase
# from rag_core.config import make_openai_client, get_settings


class MedicalBenchmarkRunner:
    """医疗评测运行器"""

    def __init__(self, service: Any, questions_path: str, corpus_path: str):
        self.service = service
        self.questions = self._load_questions(questions_path)
        self.corpus_path = corpus_path
        self.results = []

    def _load_questions(self, path: str) -> dict:
        """加载问题集"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_single_mode(self, mode: str) -> list[dict]:
        """运行单一检索模式的评测"""
        mode_results = []

        for q in self.questions["questions"]:
            question_id = q["id"]
            query_text = q["query"]
            expected_answer = q["answer"]
            query_type = q["query_type"]

            print(f"[{mode}] 评测 {question_id}: {query_text}")

            try:
                # 调用实际的查询接口
                result = self.service.query(text=query_text, mode=mode)

                # 记录结果
                mode_results.append(
                    {
                        "question_id": question_id,
                        "query": query_text,
                        "query_type": query_type,
                        "mode": mode,
                        "answer": result.answer,
                        "expected_answer": expected_answer,
                        "sources_count": len(result.sources),
                        "answer_status": result.answer_status,
                        "retrieval_status": result.retrieval_status,
                        "verification_status": result.verification_status,
                        "retries": result.retries,
                        "router_decision": (
                            result.router_decision.model_dump() if result.router_decision else None
                        ),
                        "trace_id": result.trace.trace_id if result.trace else None,
                    }
                )

            except Exception as e:
                print(f"  错误: {e}")
                mode_results.append(
                    {
                        "question_id": question_id,
                        "query": query_text,
                        "query_type": query_type,
                        "mode": mode,
                        "error": str(e),
                    }
                )

        return mode_results

    def run_all_modes(self) -> dict:
        """运行所有检索模式的评测"""
        modes = [
            "vector",
            "cypher",
            "hybrid",
            "agent_pattern",
            "agent_llm",
        ]

        all_results = {}
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"开始评测模式: {mode}")
            print(f"{'='*60}\n")
            all_results[mode] = self.run_single_mode(mode)

        return all_results

    def evaluate_with_llm_judge(self, results: dict, judge_client: Any) -> dict:
        """使用LLM评判答案正确性"""
        evaluated_results = {}

        for mode, mode_results in results.items():
            evaluated_mode_results = []

            for result in mode_results:
                if "error" in result:
                    evaluated_mode_results.append(result)
                    continue

                # LLM评判提示词
                judge_prompt = f"""你是一个医疗知识评测专家。请评估以下答案的正确性。

问题: {result['query']}

标准答案: {result['expected_answer']}

系统答案: {result['answer']}

请按照以下标准评分（1-5分）:
5分: 完全正确，信息完整准确
4分: 基本正确，有轻微遗漏或表述差异
3分: 部分正确，缺少关键信息
2分: 大部分错误，仅有少量正确信息
1分: 完全错误或答非所问

请以JSON格式返回评分和理由:
{{"score": <1-5>, "reason": "<评分理由>"}}
"""

                try:
                    response = judge_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": judge_prompt}],
                        temperature=0.0,
                    )

                    judge_result = json.loads(response.choices[0].message.content)
                    result["judge_score"] = judge_result["score"]
                    result["judge_reason"] = judge_result["reason"]
                    result["is_correct"] = judge_result["score"] >= 4

                except Exception as e:
                    print(f"  LLM评判失败: {e}")
                    result["judge_error"] = str(e)

                evaluated_mode_results.append(result)

            evaluated_results[mode] = evaluated_mode_results

        return evaluated_results

    def analyze_results(self, evaluated_results: dict) -> dict:
        """分析评测结果"""
        analysis = {
            "overall": {},
            "by_mode": {},
            "by_query_type": {},
            "routing_analysis": {},
            "self_correction_analysis": {},
        }

        # 按模式统计
        for mode, mode_results in evaluated_results.items():
            total = len(mode_results)
            correct = sum(1 for r in mode_results if r.get("is_correct", False))
            avg_retries = sum(r.get("retries", 0) for r in mode_results) / total if total > 0 else 0
            answer_statuses = {}
            verification_statuses = {}
            for r in mode_results:
                answer_status = r.get("answer_status", "unknown")
                verification_status = r.get("verification_status", "unknown")
                answer_statuses[answer_status] = answer_statuses.get(answer_status, 0) + 1
                verification_statuses[verification_status] = (
                    verification_statuses.get(verification_status, 0) + 1
                )

            analysis["by_mode"][mode] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
                "avg_retries": avg_retries,
                "answer_statuses": answer_statuses,
                "verification_statuses": verification_statuses,
            }

        # 按问题类型统计
        query_types = ["simple", "relation", "multi_hop", "global", "temporal"]
        for qtype in query_types:
            type_results = []
            for mode_results in evaluated_results.values():
                type_results.extend([r for r in mode_results if r.get("query_type") == qtype])

            if type_results:
                total = len(type_results)
                correct = sum(1 for r in type_results if r.get("is_correct", False))
                analysis["by_query_type"][qtype] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total if total > 0 else 0,
                }

        # 路由分析（仅agent模式）
        agent_modes = ["agent_pattern", "agent_llm"]
        for mode in agent_modes:
            if mode in evaluated_results:
                mode_results = evaluated_results[mode]
                routing_correct = 0
                total_with_routing = 0

                for r in mode_results:
                    if r.get("router_decision"):
                        total_with_routing += 1
                        # 这里需要根据问题的recommended_retrieval判断路由是否正确
                        # 简化处理：假设有routing_correct字段

                analysis["routing_analysis"][mode] = {
                    "total": total_with_routing,
                    "routing_correct": routing_correct,
                }

        # 自纠循环分析
        for mode, mode_results in evaluated_results.items():
            triggered = sum(1 for r in mode_results if r.get("retries", 0) > 0)
            if triggered > 0:
                success_after_retry = sum(
                    1 for r in mode_results if r.get("retries", 0) > 0 and r.get("is_correct", False)
                )
                analysis["self_correction_analysis"][mode] = {
                    "triggered": triggered,
                    "success_after_retry": success_after_retry,
                    "success_rate": success_after_retry / triggered if triggered > 0 else 0,
                }

        return analysis

    def save_results(self, results: dict, analysis: dict, output_dir: str):
        """保存评测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存原始结果
        with open(output_path / "raw_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存分析结果
        with open(output_path / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        # 生成可读报告
        self._generate_report(analysis, output_path / "report.txt")

    def _generate_report(self, analysis: dict, output_path: Path):
        """生成可读的评测报告"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("医疗领域Graph RAG评测报告\n")
            f.write("=" * 80 + "\n\n")

            # 按模式统计
            f.write("## 按检索模式统计\n\n")
            for mode, stats in analysis["by_mode"].items():
                f.write(f"### {mode}\n")
                f.write(f"  准确率: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
                f.write(f"  平均重试次数: {stats['avg_retries']:.2f}\n\n")
                f.write(f"  答案状态: {stats.get('answer_statuses', {})}\n")
                f.write(f"  验证状态: {stats.get('verification_statuses', {})}\n\n")

            # 按问题类型统计
            f.write("\n## 按问题类型统计\n\n")
            for qtype, stats in analysis["by_query_type"].items():
                f.write(f"### {qtype}\n")
                f.write(f"  准确率: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n\n")

            # 自纠循环分析
            if analysis["self_correction_analysis"]:
                f.write("\n## 自纠循环分析\n\n")
                for mode, stats in analysis["self_correction_analysis"].items():
                    f.write(f"### {mode}\n")
                    f.write(f"  触发次数: {stats['triggered']}\n")
                    f.write(f"  成功救回: {stats['success_after_retry']}\n")
                    f.write(f"  成功率: {stats['success_rate']:.2%}\n\n")


def main():
    """主函数示例"""
    # 1. 初始化服务（需要根据实际项目调整）
    # cfg = get_settings()
    # driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    # client = make_openai_client(cfg)
    # service = PipelineService(driver, client)

    # 2. 创建评测运行器
    # runner = MedicalBenchmarkRunner(
    #     service=service,
    #     questions_path="test/medical_benchmark/questions_master.json",
    #     corpus_path="test/medical_benchmark/corpus_medical.txt"
    # )

    # 3. 运行评测
    # results = runner.run_all_modes()

    # 4. LLM评判
    # evaluated_results = runner.evaluate_with_llm_judge(results, client)

    # 5. 分析结果
    # analysis = runner.analyze_results(evaluated_results)

    # 6. 保存结果
    # runner.save_results(evaluated_results, analysis, "test/medical_benchmark/results")

    print("评测模板已创建，请根据实际项目调整代码")


if __name__ == "__main__":
    main()
