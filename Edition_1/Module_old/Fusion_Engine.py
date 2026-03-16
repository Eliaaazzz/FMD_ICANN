import json
import numpy as np
from datetime import datetime


class EvidenceFusionLayer:
    def __init__(self, client, model="qwen-max"): # 修改此处
        self.client = client
        self.model = model

        # 定义工具的“证据权重” (Evidence Weight)
        # 事实类工具权重最高，修辞类权重较低
        self.TOOL_WEIGHTS = {
            "FCV": 0.5,  # 事实核查 (最关键)
            "TLV": 0.3,  # 时间逻辑
            "SCP": 0.3,  # 来源可信度
            "EVA": 0.2,  # 常识违背
            "CGT": 0.2,  # 上下文缺失
            "PID": 0.1,  # 情感煽动 (辅助)
            "RMD": 0.1  # 营销词汇 (辅助)
        }

    def _normalize_tool_output(self, tool_name, output):
        """
        将不同工具的异构输出归一化为证据质量分数 (0.0 - 1.0)
        1.0 = 支持“是真新闻”
        0.0 = 支持“是假新闻”
        0.5 = 不确定
        """
        score = 0.5  # 默认中立

        try:
            if tool_name == "SCP":  # 来源
                score = output.get("credibility_score", 0.5)

            elif tool_name == "FCV":  # 事实
                # 如果有一致性分数直接用，否则看矛盾数量
                if "consistency_score" in output:
                    score = output["consistency_score"]
                elif output.get("contradictions"):
                    score = 0.1  # 有矛盾，低分
                else:
                    score = 0.8  # 无矛盾，高分

            elif tool_name == "TLV":  # 时间
                if output.get("is_chronologically_valid") is True:
                    score = 0.9
                else:
                    score = 0.1

            elif tool_name == "PID" or tool_name == "RMD":  # 修辞/营销
                # 情感强度越高，可信度越低 (反比)
                intensity = output.get("emotional_intensity", 0.0) or output.get("hype_score", 0.0)
                score = 1.0 - intensity

            elif tool_name == "CGT":  # 上下文
                score = output.get("alignment_score", 0.5)

            elif tool_name == "EVA":  # 常识
                if output.get("violation_detected") is True:
                    score = 0.1
                else:
                    score = 0.9

        except Exception as e:
            print(f"[Warning] Normalization failed for {tool_name}: {e}")

        return float(score)

    def _calculate_ds_fusion(self, tool_outputs):
        """
        [算法核心] 简化的 D-S 证据融合算法。
        计算：Belief (置信度), Plausibility (似然度), Uncertainty (不确定性)
        """
        mass_real = 0.0
        mass_fake = 0.0
        total_weight = 0.0

        for tool_name, result in tool_outputs.items():
            if tool_name in ["CPE"]: continue  # 跳过之前的总结工具

            weight = self.TOOL_WEIGHTS.get(tool_name, 0.1)
            score = self._normalize_tool_output(tool_name, result)

            # 简单的加权累积 (D-S 的简化实现，避免复杂的正交和计算)
            # score > 0.5 支持 Real, score < 0.5 支持 Fake
            if score > 0.6:
                mass_real += weight * (score - 0.5) * 2  # 归一化强度
            elif score < 0.4:
                mass_fake += weight * (0.5 - score) * 2

            total_weight += weight

        # 归一化 Mass
        if total_weight > 0:
            mass_real /= total_weight
            mass_fake /= total_weight

        # 修正：防止总和超过1
        total_mass = mass_real + mass_fake
        uncertainty = 1.0 - total_mass if total_mass < 1.0 else 0.0

        return {
            "mass_real": round(mass_real, 3),
            "mass_fake": round(mass_fake, 3),
            "uncertainty": round(uncertainty, 3),
            "conflict_level": round(min(mass_real, mass_fake), 3)  # 如果两者都很高，说明冲突严重
        }

    def generate_final_report(self, context_data):
        """
        生成最终的可解释性报告
        """
        text = context_data.get('initial_text')
        tool_outputs = context_data.get('tool_outputs', {})

        # 1. 计算数学指标
        ds_metrics = self._calculate_ds_fusion(tool_outputs)

        # 2. 准备 Prompt 上下文
        evidence_summary = json.dumps(tool_outputs, indent=2)
        metrics_str = json.dumps(ds_metrics, indent=2)

        prompt = f"""
        TASK: Synthesize the Final Explainable Verdict for the detected news.

        TARGET TEXT: "{text}"

        TOOL EVIDENCE TRACE:
        {evidence_summary}

        CALCULATED METRICS (Dempster-Shafer Fusion):
        {metrics_str}

        INSTRUCTIONS:
        1. Base your verdict on the Evidence Trace. Use the Metrics to resolve conflicts (e.g., if Mass_Fake > Mass_Real, lean towards Fake).
        2. IF "conflict_level" is high (e.g. > 0.2), explicitly explain WHY (e.g. "Source is good but Facts are wrong").
        3. Structure the explanation as a logic path.

        OUTPUT JSON:
        {{
            "final_label": "Real News / Fake News / Misleading / Unverified",
            "confidence_score": 0.0 to 1.0,
            "risk_level": "Low / Medium / High / Critical",
            "explanation_path": "Tool A found X -> Tool B confirmed Y -> Conclusion Z",
            "key_contradictions": ["List strictly factual contradictions found"],
            "suggestion": "Advice for the user (e.g. 'Wait for SEC filing')"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are the Final Adjudicator of the CATO framework. Output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            # 使用之前的 robust parser (假设你已经导入)
            # 这里简单处理
            content = response.choices[0].message.content
            # 简单的 JSON 提取
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                final_json = json.loads(match.group(0))
            else:
                final_json = json.loads(content)

            # 将数学指标也合并进去，供前端展示
            final_json["ds_metrics"] = ds_metrics
            return final_json

        except Exception as e:
            return {"error": str(e), "verdict": "Unverified"}