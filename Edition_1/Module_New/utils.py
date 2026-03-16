import json
import re


def robust_json_parse(llm_output):
    """
    鲁棒的JSON解析器，专门处理LLM输出的各种不规范格式。
    """
    try:
        # 1. 尝试直接解析
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    # 2. 尝试提取 ```json ... ``` 内部的内容
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 尝试寻找最外层的 {}
    pattern = r"\{.*\}"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. 如果是Python的字典字符串（单引号），尝试转换
    try:
        # 注意：eval有安全风险，但在受控的prompt环境下通常用于最后的兜底
        # 更好的方式是使用 ast.literal_eval
        import ast
        return ast.literal_eval(llm_output)
    except:
        pass

    print(f"Error: Failed to parse JSON from output: {llm_output[:100]}...")
    return None