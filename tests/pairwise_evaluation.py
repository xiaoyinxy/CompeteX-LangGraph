from langchain_anthropic import ChatAnthropic
from langsmith.evaluation import evaluate_comparative
from pydantic import BaseModel, Field

HEAD_TO_HEAD_PROMPT = """
我们正在测试 Deep Research Agent 的两种不同实现。该 Agent 用于围绕给定问题开展深度研究。

研究问题：
{question}

第一种实现的回答：
{answer_a}

第二种实现的回答：
{answer_b}

评估这些 Agent 时，请考虑以下标准：
- 优秀的 Research Agent 应研究足够多的来源来回答问题。来源应当多样且质量较高。来源并非越多越好，但数量和质量必须足以使报告主张可信；
- 应完整、全面地回答用户问题；
- Deep Research Agent 的运行成本较高，因此用户期待高质量且足够详细的答案。用户通常应能从回答中获得所需信息，而不必继续追问；
- 所有主张都应提供引用，且引用格式应当易于阅读和理解。

重要：
两种实现采用了不同的研究方式，因此可能获得不同的信息和来源，这是需要重点评估的差异。
请判断哪个 Agent 找到了更好的来源，并更好地回答了问题。最重要的评估标准是答案质量和完整性。

请根据以上标准选择更好的回答，并详细说明原因。
"""

class HeadToHeadRanking(BaseModel):
    reasoning: str = Field(description="选择更优回答的原因，需要给出详细说明。")
    preferred_answer: int = Field(description="在 1 和 2 中选择更优回答；1 表示第一个回答，2 表示第二个回答。")


def head_to_head_evaluator(inputs: dict, outputs: list[dict]) -> list:
    grader_llm = ChatAnthropic(
        model="claude-opus-4-20250514",
        max_tokens=20000,
        thinking={"type": "enabled", "budget_tokens": 16000},
    )

    response = grader_llm.with_structured_output(HeadToHeadRanking).invoke(HEAD_TO_HEAD_PROMPT.format(
        question=inputs["messages"][0]["content"],
        answer_a=outputs[0].get("final_report", "N/A"),
        answer_b=outputs[1].get("final_report", "N/A"),
    ))

    if response.preferred_answer == 1:
        scores = [1, 0]
    elif response.preferred_answer == 2:
        scores = [0, 1]
    else:
        scores = [0, 0]
    return scores


ALL_THREE_PROMPT = """
我们正在测试 Deep Research Agent 的三种不同实现。该 Agent 用于围绕给定问题开展深度研究。

研究问题：
{question}

第一种实现的回答：
{answer_a}

第二种实现的回答：
{answer_b}

第三种实现的回答：
{answer_c}

评估这些 Agent 时，请考虑以下标准：
- 优秀的 Research Agent 应研究足够多的来源来回答问题。来源应当多样且质量较高。来源并非越多越好，但必须足以使报告主张可信；
- 应完整、全面地回答用户问题；
- Deep Research Agent 的运行成本较高，因此用户期待高质量且足够详细的答案，并且通常不必继续追问；
- 所有主张都应提供引用，且引用格式应当易于阅读和理解。

重要：
三种实现采用了不同的研究方式，因此可能获得不同的信息和来源，这是需要重点评估的差异。
请判断哪个 Agent 找到了更好的来源，并更好地回答了问题。最重要的评估标准是答案质量和完整性。

请根据以上标准对三个回答进行排名：1 表示最佳，2 表示第二，3 表示最差。
请详细说明采用该排名的原因。
"""

class Rankings(BaseModel):
    reasoning: str = Field(description="采用当前排名的原因，需要给出详细说明。")
    preferred_answer: int = Field(description="最佳回答的编号；1、2、3 分别表示第一、第二、第三个回答。")
    second_best_answer: int = Field(description="第二名回答的编号；1、2、3 分别表示第一、第二、第三个回答。")
    worst_answer: int = Field(description="最差回答的编号；1、2、3 分别表示第一、第二、第三个回答。")

def free_for_all_evaluator(inputs: dict, outputs: list[dict]) -> list:
    grader_llm = ChatAnthropic(
        model="claude-opus-4-20250514",
        max_tokens=20000,
        thinking={"type": "enabled", "budget_tokens": 16000},
    )

    response = grader_llm.with_structured_output(Rankings).invoke(ALL_THREE_PROMPT.format(
        question=inputs["messages"][0]["content"],
        answer_a=outputs[0].get("final_report", "N/A"),
        answer_b=outputs[1].get("final_report", "N/A"),
        answer_c=outputs[2].get("final_report", "N/A"),
    ))

    scores = [0, 0, 0]
    scores[response.preferred_answer - 1] = 1
    scores[response.second_best_answer - 1] = .5
    scores[response.worst_answer - 1] = 0
    return scores

single_agent = "DR Single Agent - Tavily #-87e8a6c0"
multi_agent_supervisor = "DR Supervisor: Multi Agent - Tavily #-cd25e7e3"
multi_agent_supervisor_v2 = "DR Supervisor: Multi Agent - Tavily (v2) #-40967f53"
multi_agent_workflow = "DR MAW - Tavily #-c6818a83"


# evaluate_comparative(
#     (single_agent_experiment_name, multi_agent_supervisor_experiment_name, multi_agent_workflow_experiment_name),  # 替换为需要比较的实验名称或 ID
#     evaluators=[free_for_all_evaluator],
#     randomize_order=True,
# )

evaluate_comparative(
    (single_agent, multi_agent_supervisor_v2),  # 替换为需要比较的实验名称或 ID
    evaluators=[head_to_head_evaluator],
    randomize_order=True,
)
