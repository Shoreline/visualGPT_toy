# orchestrator.py
import json, os
from tools import TOOLS, run_tool
from rich import print
from openai import OpenAI  # 如果你用 OpenAI；本地模型请替换

SYSTEM_PROMPT_PREFIX = """你是一个视觉工具编排助手。你可以使用下列工具：
"""

SYSTEM_PROMPT_SUFFIX = """
规则：
1) 每一步要么输出一个 JSON 行为： {"action": "<tool_name>", "args": {...}}
2) 要么输出最终答案： {"final": "<自然语言回答>"} 
3) 工具返回会给你 {"image_path": "...", "summary": "..."}。如果产生了新图，请在后续步骤使用这个最新 image_path。
4) 连续多步，直到可以给出最终答案。避免无意义循环。"""



def llm_call(messages, tools, model="gpt-4o-mini"):
    from openai import OpenAI
    tool_list_txt = "\n".join(
        [f"- {name}: {spec.description} | schema={spec.args_schema}" for name, spec in tools.items()]
    )
    system = SYSTEM_PROMPT_PREFIX + tool_list_txt + SYSTEM_PROMPT_SUFFIX

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def step_loop(user_goal: str, image_path: str, max_steps=8):
    messages = [{"role":"user","content": f"{user_goal}\n初始图像: {image_path}"}]
    last_image = image_path
    for _ in range(max_steps):
        out = llm_call(messages, TOOLS)
        print("[cyan]LLM:[/cyan]", out)
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            messages.append({"role":"assistant","content": out})
            # 引导它用 JSON
            messages.append({"role":"user","content": "请用严格 JSON 输出动作或最终答案。"})
            continue
        if "final" in obj:
            print("[bold green]FINAL:[/bold green]", obj["final"])
            print("[bold yellow]Result image:[/bold yellow]", last_image)
            return
        if "action" in obj and "args" in obj:
            args = obj["args"]
            # 自动把 {LAST_IMAGE} 占位替换为最新图
            for k, v in list(args.items()):
                if isinstance(v, str) and v == "{LAST_IMAGE}":
                    args[k] = last_image
            if "image_path" not in args and "load_image" != obj["action"]:
                args["image_path"] = last_image
            res = run_tool(obj["action"], **args)
            last_image = res.get("image_path", last_image)
            tool_feedback = json.dumps({"tool_result": res["summary"], "image_path": last_image}, ensure_ascii=False)
            messages.append({"role":"assistant","content": out})
            messages.append({"role":"user","content": f"工具已执行：{tool_feedback}。请基于最新结果继续。"})
        else:
            messages.append({"role":"assistant","content": out})
            messages.append({"role":"user","content":"动作缺少 action/args，请重试并仅输出 JSON。"})
    print("[red]达到最大步数，提前结束。[/red]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Visual ChatGPT orchestrator.")
    parser.add_argument("goal", type=str, help="用户的任务目标描述")
    parser.add_argument("--image", type=str, default="cat.jpg", help="初始图像路径")
    args = parser.parse_args()

    step_loop(args.goal, args.image)
