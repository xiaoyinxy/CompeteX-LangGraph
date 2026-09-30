"""Deep Research Agent 的 LangGraph 主实现。"""

import asyncio
from typing import Literal
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    StructuredResearchResult,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化整个 Agent 工作流共用的可配置模型
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "extra_body"),
)


# 竞品分析的五个必研维度。首次研究由程序生成这五张工单，避免依赖模型自由拆解而漏项。
REQUIRED_COMPETITOR_DIMENSIONS = (
    "产品定位",
    "目标用户",
    "核心能力",
    "商业模式",
    "近期动态",
)


def build_required_research_tool_calls(research_brief: str) -> list[dict]:
    """为竞品分析创建五个可并行执行的标准研究工具调用。"""
    dimension_guidance = {
        "产品定位": "研究各对象解决的核心问题、价值主张、使用场景、市场类别、差异化定位和官方表述。",
        "目标用户": "研究各对象的核心用户群、典型角色、用户需求、使用门槛、覆盖市场及可验证的用户规模或画像信号。",
        "核心能力": "研究各对象的关键功能、模型或技术能力、产品体验、生态集成、优势短板，并基于相同指标横向比较。",
        "商业模式": "研究各对象的免费与付费策略、定价、收入来源、获客与分发渠道、企业合作及商业化成熟度。",
        "近期动态": "重点检索最近十二个月的中文互联网信息，研究产品发布、版本更新、融资合作、用户数据、政策影响和市场动作；必须标注事件日期。",
    }

    calls = []
    for dimension in REQUIRED_COMPETITOR_DIMENSIONS:
        research_topic = (
            f"总体研究任务：{research_brief}\n\n"
            f"本 Researcher 只负责【{dimension}】维度。{dimension_guidance[dimension]}"
            "优先检索中文官方网站、官方账号、权威媒体和可靠行业资料，并使用中文关键词搜索；"
            "所有事实必须保留标题、URL、发布日期（如可获得）和证据内容。"
            "输出时只陈述本维度结论，明确列出关键发现、横向对比、证据置信度和证据缺口，不得扩展到其他维度。"
        )
        calls.append({
            "name": "ConductResearch",
            "args": {
                "research_dimension": dimension,
                "research_topic": research_topic,
            },
            "id": f"required_{uuid4().hex}",
            "type": "tool_call",
        })
    return calls


def format_structured_research_result(
    result: StructuredResearchResult,
    required_dimension: str,
) -> str:
    """把结构化 Researcher 结果渲染为便于 Supervisor 阅读的统一 Markdown。"""
    lines = [
        f"## 【{required_dimension}】",
        "",
        "### 结论摘要",
        result.summary.strip() or "公开证据不足。",
        "",
        "### 关键发现",
    ]
    lines.extend(f"- {item}" for item in result.key_findings)
    if not result.key_findings:
        lines.append("- 公开证据不足。")

    lines.extend(["", "### 横向对比"])
    lines.extend(f"- {item}" for item in result.comparison_points)
    if not result.comparison_points:
        lines.append("- 暂无可验证的横向对比结论。")

    lines.extend(["", "### 结构化证据"])
    for item in result.evidence:
        source = f"[{item.source_title}]({item.url})" if item.url.startswith(("http://", "https://")) else item.source_title
        lines.append(
            f"- {source}｜日期：{item.published_at}｜置信度：{item.confidence}｜证据：{item.evidence}"
        )
    if not result.evidence:
        lines.append("- 公开证据不足。")

    lines.extend(["", "### 证据缺口"])
    lines.extend(f"- {item}" for item in result.evidence_gaps)
    if not result.evidence_gaps:
        lines.append("- 未发现需要特别说明的证据缺口。")

    lines.extend(["", "### 来源"])
    seen_urls = set()
    source_index = 1
    for item in result.evidence:
        if item.url.startswith(("http://", "https://")) and item.url not in seen_urls:
            lines.append(f"[{source_index}] {item.source_title}：{item.url}")
            seen_urls.add(item.url)
            source_index += 1
    if not seen_urls:
        lines.append("公开链接缺失。")

    return "\n".join(lines)


def format_research_failure(dimension: str, error: BaseException) -> str:
    """把单个 Researcher 异常转换成可被 Supervisor 识别的结构化缺口。"""
    return "\n".join([
        f"## 【{dimension}】",
        "",
        "### 结论摘要",
        "本维度研究执行失败，未生成可验证结论。",
        "",
        "### 关键发现",
        "- 公开证据不足。",
        "",
        "### 横向对比",
        "- 暂无可验证的横向对比结论。",
        "",
        "### 结构化证据",
        "- 公开证据不足。",
        "",
        "### 证据缺口",
        f"- Researcher 执行异常：{type(error).__name__}：{error}",
        "",
        "### 来源",
        "公开链接缺失。",
    ])


def deepseek_compat(model_name: str) -> dict:
    """当 LangChain 强制执行结构化输出或工具调用时，关闭 DeepSeek V4 的 thinking 模式。"""
    if str(model_name).lower().startswith("deepseek:"):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息，并在研究范围不明确时提出澄清问题。

    如果关闭了澄清能力，或者现有信息已经足够，则直接进入研究流程。

    参数：
        state: 包含用户消息的当前 Agent State
        config: 包含模型设置和偏好的运行配置

    返回：
        提出澄清问题并结束，或继续生成 Research Brief 的 Command
    """
    # 步骤 1：检查配置是否启用了澄清能力
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清节点，直接进入研究流程
        return Command(goto="write_research_brief")
    
    # 步骤 2：准备执行结构化澄清分析的模型
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
        **deepseek_compat(configurable.research_model),
    }
    
    # 为模型配置结构化输出与重试逻辑
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # 步骤 3：分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 步骤 4：根据澄清结果选择后续路径
    if response.need_clarification:
        # 向用户提出澄清问题并结束当前 Run
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 返回确认消息并继续研究
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化 Research Brief，并初始化 Supervisor。

    此函数生成用于指导 Supervisor 的研究简报，同时通过对应 Prompt 和指令
    初始化 Supervisor 上下文。

    参数：
        state: 包含用户消息的当前 Agent State
        config: 包含模型设置的运行配置

    返回：
        携带初始化上下文并进入 Research Supervisor 的 Command
    """
    # 步骤 1：配置用于结构化输出的研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
        **deepseek_compat(configurable.research_model),
    }
    
    # 配置模型以生成结构化研究问题
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 步骤 2：根据用户消息生成结构化 Research Brief
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 步骤 3：使用 Research Brief 和研究指令初始化 Supervisor
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """规划研究策略并向 Researcher 委派任务的研究主管节点。

    Supervisor 分析 Research Brief，并决定如何将研究拆分为可执行任务。
    它可以使用 think_tool 规划策略，使用 ConductResearch 委派任务，
    或在研究结果充分时调用 ResearchComplete。

    参数：
        state: 包含消息和研究上下文的当前 Supervisor State
        config: 包含模型设置的运行配置

    返回：
        进入 supervisor_tools 执行工具的 Command
    """
    # 步骤 1：使用可用工具配置 Supervisor 模型
    configurable = Configuration.from_runnable_config(config)

    # 首轮不依赖模型自由拆解，固定创建五个标准维度的并行研究工单。
    if state.get("research_iterations", 0) == 0:
        required_calls = build_required_research_tool_calls(state.get("research_brief", ""))
        response = AIMessage(
            content="已按竞品分析标准创建产品定位、目标用户、核心能力、商业模式和近期动态五个并行研究任务。",
            tool_calls=required_calls,
        )
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "research_iterations": 1,
            },
        )

    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
        **deepseek_compat(configurable.research_model),
    }
    
    # 可用工具：研究委派、完成信号和策略反思
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # 为模型绑定工具、重试逻辑和运行参数
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 步骤 2：根据当前上下文生成 Supervisor 响应
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # 步骤 3：更新 State 并进入工具执行节点
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行 Supervisor 调用的工具，包括研究委派和策略反思。

    此函数处理三类 Supervisor 工具调用：
    1. think_tool——执行策略反思并继续当前循环；
    2. ConductResearch——把研究任务委派给子 Researcher；
    3. ResearchComplete——表示研究阶段已经完成。

    参数：
        state: 包含消息和迭代次数的当前 Supervisor State
        config: 包含研究限制和模型设置的运行配置

    返回：
        继续 Supervisor 循环或结束研究阶段的 Command
    """
    # 步骤 1：提取当前 State 并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # 定义研究阶段的退出条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # 满足任意终止条件时退出研究阶段
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # 步骤 2：统一处理所有工具调用（think_tool 和 ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # 处理 think_tool 调用（策略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"已记录研究反思：{reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # 处理 ConductResearch 调用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # 五个标准维度必须同批并行；额外研究任务仍受配置上限控制。
            effective_concurrency = max(
                configurable.max_concurrent_research_units,
                len(REQUIRED_COMPETITOR_DIMENSIONS),
            )
            allowed_conduct_research_calls = conduct_research_calls[:effective_concurrency]
            overflow_conduct_research_calls = conduct_research_calls[effective_concurrency:]
            
            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                    "research_dimension": tool_call["args"].get("research_dimension", "补充研究"),
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            # 单个 Researcher 失败时保留其他分支结果，避免一处异常使整批研究丢失。
            tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # 使用研究结果创建 ToolMessage
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                dimension = tool_call["args"].get("research_dimension", "补充研究")
                if isinstance(observation, BaseException):
                    content = format_research_failure(dimension, observation)
                else:
                    content = observation.get(
                        "compressed_research",
                        format_research_failure(dimension, RuntimeError("结构化研究结果缺失")),
                    )
                all_tool_messages.append(ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # 为超出并发上限的研究调用生成错误消息
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"错误：研究单元数量已超过最大并发上限，本研究任务未执行。请将并发研究单元控制在 {configurable.max_concurrent_research_units} 个以内。",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # 汇总全部研究结果中的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
                if isinstance(observation, dict)
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # 处理研究执行异常
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token 超限或发生其他异常时结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # 步骤 3：返回包含全部工具结果的 Command
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor 子图构建
# 创建负责研究委派与协调的 Supervisor 工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加负责研究管理的 Supervisor 节点
supervisor_builder.add_node("supervisor", supervisor)           # Supervisor 主逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行处理器

# 定义 Supervisor 工作流的边
supervisor_builder.add_edge(START, "supervisor")  # Supervisor 入口

# 编译 Supervisor 子图，供主工作流调用
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """围绕具体主题开展聚焦研究的单个 Researcher 节点。

    Researcher 接收 Supervisor 分配的具体研究主题，并使用搜索、think_tool、
    MCP 等可用工具全面收集信息。它可以在搜索之间使用 think_tool 规划策略。

    参数：
        state: 包含消息和主题上下文的当前 Researcher State
        config: 包含模型设置和工具可用性的运行配置

    返回：
        进入 researcher_tools 执行工具的 Command
    """
    # 步骤 1：加载配置并验证工具是否可用
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # 获取全部可用研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "没有可用于研究的工具：请配置 Search API，或在配置中添加 MCP 工具。"
        )
    
    # 步骤 2：使用工具配置 Researcher 模型
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
        **deepseek_compat(configurable.research_model),
    }
    
    # 如果存在 MCP 上下文，则将其加入系统 Prompt
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # 为模型绑定工具、重试逻辑和运行参数
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 步骤 3：结合系统上下文生成 Researcher 响应
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # 步骤 4：更新 State 并进入工具执行节点
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """执行工具并安全处理异常。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"工具执行失败：{str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行 Researcher 调用的工具，包括搜索工具和策略反思。

    此函数处理多种 Researcher 工具调用：
    1. think_tool——执行策略反思并继续研究对话；
    2. 搜索工具（tavily_search、web_search）——收集信息；
    3. MCP 工具——调用外部集成；
    4. ResearchComplete——表示单项研究任务已经完成。

    参数：
        state: 包含消息和迭代次数的当前 Researcher State
        config: 包含研究限制和工具设置的运行配置

    返回：
        继续研究循环或进入压缩节点的 Command
    """
    # 步骤 1：提取当前 State 并检查提前退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 没有发生工具调用（包括原生 Web Search）时提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # 步骤 2：处理其他工具调用（搜索、MCP 工具等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # 并行执行全部工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # 根据执行结果创建 ToolMessage
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # 步骤 3：处理工具后检查退出条件
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩节点
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # 携带工具结果继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究发现压缩、整理为简洁的结构化摘要。

    此函数接收 Researcher 产生的全部研究发现、工具输出和 AI 消息，
    在保留重要信息与发现的同时，将其整理成清晰、完整的摘要。

    参数：
        state: 已经积累研究消息的当前 Researcher State
        config: 包含压缩模型设置的运行配置

    返回：
        包含压缩研究摘要和原始笔记的字典
    """
    # 步骤 1：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = (
        configurable_model
        .with_structured_output(StructuredResearchResult)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config({
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.compression_model, config),
            "tags": ["langsmith:nostream"],
            **deepseek_compat(configurable.compression_model),
        })
    )

    # 步骤 2：准备需要压缩的消息
    researcher_messages = list(state.get("researcher_messages", []))
    research_dimension = state.get("research_dimension", "补充研究")
    
    # 添加从研究模式切换到压缩模式的指令
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message.format(
        research_dimension=research_dimension,
    )))
    
    # 步骤 3：执行压缩，并针对 Token 超限问题进行重试
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # 创建专门用于压缩任务的系统 Prompt
            compression_prompt = compress_research_system_prompt.format(
                date=get_today_str(),
                research_dimension=research_dimension,
            )
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)
            
            # 从全部工具消息和 AI 消息中提取原始笔记
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # 返回成功的压缩结果
            return {
                "compressed_research": format_structured_research_result(response, research_dimension),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Token 超限时删除较早的消息后重试
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # 其他异常继续重试
            continue
    
    # 步骤 4：全部尝试失败后返回错误结果
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "研究报告整理失败：已超过最大重试次数",
        "raw_notes": [raw_notes_content]
    }

# Researcher 子图构建
# 创建围绕具体主题开展聚焦研究的单个 Researcher 工作流
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# 添加研究执行与压缩节点
researcher_builder.add_node("researcher", researcher)                 # Researcher 主逻辑
researcher_builder.add_node("researcher_tools", researcher_tools)     # 工具执行处理器
researcher_builder.add_node("compress_research", compress_research)   # 研究材料压缩

# 定义 Researcher 工作流的边
researcher_builder.add_edge(START, "researcher")           # Researcher 入口
researcher_builder.add_edge("compress_research", END)      # 压缩完成后的出口

# 编译 Researcher 子图，供 Supervisor 并行调用
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成完整的最终研究报告，并在 Token 超限时进行重试。

    此函数接收全部研究发现，并使用配置的报告模型将其整理为结构清晰、
    内容完整的最终报告。

    参数：
        state: 包含研究发现和上下文的 Agent State
        config: 包含模型设置和 API Key 的运行配置

    返回：
        包含最终报告和清理后 State 的字典
    """
    # 步骤 1：提取研究发现并准备清理 State
    notes = state.get("notes", []) or state.get("raw_notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # 步骤 2：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
        **deepseek_compat(configurable.final_report_model),
    }
    
    # 步骤 3：生成报告，并在 Token 超限时重试
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # 使用全部研究上下文创建完整 Prompt
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # 返回成功生成的报告
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # 通过逐步截断内容处理 Token 超限错误
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # 第一次重试：确定初始截断上限
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"最终报告生成失败：Token 已超限，但无法确定该模型的最大上下文长度。请在 open_deep_research/utils.py 的模型映射中补充相关信息。{e}",
                            "messages": [AIMessage(content="报告因 Token 超限而生成失败")],
                            **cleared_state
                        }
                    # 使用 Token 上限的 4 倍作为字符截断上限的近似值
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续每次重试将截断上限减少 10%
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # 截断研究发现后重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 遇到非 Token 超限异常时立即返回错误
                return {
                    "final_report": f"最终报告生成失败：{e}",
                    "messages": [AIMessage(content="报告因运行异常而生成失败")],
                    **cleared_state
                }
    
    # 步骤 4：所有重试机会用尽后返回失败结果
    return {
        "final_report": "最终报告生成失败：已超过最大重试次数",
        "messages": [AIMessage(content="报告在达到最大重试次数后仍生成失败")],
        **cleared_state
    }

# Deep Researcher 主图构建
# 创建从用户输入到最终报告的完整 Deep Research 工作流
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# 添加完整研究流程的主工作流节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 用户需求澄清阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成阶段

# 定义主工作流顺序执行的边
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # 主流程入口
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # 从研究进入报告生成
deep_researcher_builder.add_edge("final_report_generation", END)                   # 最终出口

# 编译完整的 Deep Researcher 工作流
deep_researcher = deep_researcher_builder.compile()
