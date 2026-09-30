"""Deep Research Agent 的图状态定义与数据结构。"""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出
###################
class ConductResearch(BaseModel):
    """调用此工具研究一个具体主题。"""
    research_dimension: Literal[
        "产品定位",
        "目标用户",
        "核心能力",
        "商业模式",
        "近期动态",
        "补充研究",
    ] = Field(
        description="本次研究所属的固定竞品分析维度；不属于五个标准维度的补充任务使用“补充研究”。",
    )
    research_topic: str = Field(
        description="需要研究的主题。应当只包含一个主题，并使用至少一个自然段进行详细描述。",
    )

class ResearchComplete(BaseModel):
    """调用此工具表示研究已经完成。"""

class Summary(BaseModel):
    """包含关键发现的研究摘要。"""
    
    summary: str
    key_excerpts: str


class ResearchEvidence(BaseModel):
    """单条研究证据的结构化表示。"""

    source_title: str = Field(description="来源页面或材料的标题。")
    url: str = Field(description="可访问的来源 URL；确实没有 URL 时填写“公开链接缺失”。")
    published_at: str = Field(description="来源发布日期；无法确认时填写“日期未知”。")
    evidence: str = Field(description="该来源支持的事实或判断，不得超出来源内容。")
    confidence: Literal["高", "中", "低"] = Field(description="根据来源权威性、时效性和交叉验证情况判断的置信度。")


class StructuredResearchResult(BaseModel):
    """单个 Researcher 提交给 Supervisor 的结构化研究结果。"""

    research_dimension: Literal[
        "产品定位",
        "目标用户",
        "核心能力",
        "商业模式",
        "近期动态",
        "补充研究",
    ] = Field(description="本次研究对应的竞品分析维度。")
    summary: str = Field(description="该维度的核心结论摘要。")
    key_findings: list[str] = Field(description="有证据支持的关键发现列表。")
    comparison_points: list[str] = Field(description="不同竞品之间可直接比较的观察列表。")
    evidence: list[ResearchEvidence] = Field(description="支撑结论的结构化证据列表。")
    evidence_gaps: list[str] = Field(description="尚无可靠公开证据、存在冲突或需要进一步验证的信息。")

class ClarifyWithUser(BaseModel):
    """用户澄清请求的数据模型。"""
    
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="用于请用户澄清报告范围的问题。",
    )
    verification: str = Field(
        description="确认用户提供必要信息后将开始研究的消息。",
    )

class ResearchQuestion(BaseModel):
    """用于指导研究的研究问题与简报。"""
    
    research_brief: str = Field(
        description="用于指导后续研究的研究问题。",
    )


###################
# State 定义
###################

def override_reducer(current_value, new_value):
    """允许覆盖 State 中现有值的 Reducer 函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """输入 State，只包含 `messages`。"""

class AgentState(MessagesState):
    """主 Agent State，包含消息和研究数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """Supervisor 用于管理研究任务的 State。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """单个 Researcher 执行研究时使用的 State。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    research_dimension: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """单个 Researcher 的输出 State。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
