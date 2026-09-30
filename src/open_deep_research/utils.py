"""Deep Research Agent 使用的工具函数与辅助方法。"""

import asyncio
import logging
import os
import re
import threading
import warnings
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
from xml.etree import ElementTree

import aiohttp
import httpx
from bs4 import BeautifulSoup
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily 搜索工具
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "面向完整、准确和可信结果进行优化的搜索引擎，"
    "适合用于回答与近期事件有关的问题。"
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """通过 Tavily Search API 获取并总结搜索结果。

    参数：
        queries: 需要执行的搜索查询列表
        max_results: 每个查询最多返回的结果数
        topic: 搜索结果的主题过滤条件（general、news 或 finance）
        config: 包含 API Key 和模型设置的运行配置

    返回：
        包含搜索结果摘要的格式化字符串
    """
    # 步骤 1：异步执行搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # 步骤 2：按 URL 去重，避免重复处理相同内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # 步骤 3：根据配置初始化总结模型
    configurable = Configuration.from_runnable_config(config)
    
    # 可配置的字符上限，用于避免超过模型 Token 限制
    max_char_to_include = configurable.max_content_length
    
    # 初始化带重试逻辑的总结模型
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"],
        **(
            {"extra_body": {"thinking": {"type": "disabled"}}}
            if configurable.summarization_model.lower().startswith("deepseek:")
            else {}
        ),
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # 步骤 4：创建总结任务，跳过原始内容为空的结果
    async def noop():
        """处理没有原始内容的结果，不执行任何操作。"""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    
    # 步骤 5：并行执行全部总结任务
    summaries = await asyncio.gather(*summarization_tasks)
    
    # 步骤 6：将搜索结果与对应摘要合并
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # 步骤 7：格式化最终输出
    if not summarized_results:
        return "没有找到有效的搜索结果。请尝试其他搜索关键词或更换 Search API。"
    
    formatted_output = "搜索结果：\n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- 来源 {i+1}：{result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"摘要：\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output


CHINESE_SEARCH_CONCURRENCY = threading.BoundedSemaphore(2)
CHINESE_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
        "Chrome/126.0 Mobile Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def parse_360_search_results(html: str, max_results: int) -> list[dict[str, str]]:
    """解析 360 搜索结果页，优先返回页面提供的来源直达链接。"""
    soup = BeautifulSoup(html, "html.parser")
    parsed_results: list[dict[str, str]] = []

    for item in soup.select("li.res-list, div.g-card.res-list"):
        link_node = item.select_one("h3.res-title a")
        title_node = link_node or item.select_one("h3.res-title")
        if not title_node:
            continue

        title = title_node.get_text(" ", strip=True)
        link_node = link_node or title_node.find_parent("a")
        direct_url = (
            title_node.get("data-mdurl")
            or item.get("data-pcurl")
            or (link_node.get("href") if link_node else "")
            or ""
        )
        if not title or not direct_url:
            continue

        summary_node = item.select_one(".res-list-summary, .res-desc, p.summary, .summary")
        date_node = item.select_one("time, .g-c-gray")
        parsed_results.append({
            "title": title,
            "href": direct_url,
            "body": summary_node.get_text(" ", strip=True) if summary_node else "未提供摘要",
            "date": (
                date_node.get_text(" ", strip=True).replace("\xa0", " ").strip(" -")
                if date_node else "日期未知"
            ),
        })
        if len(parsed_results) >= max_results:
            break

    return parsed_results


def parse_google_news_results(xml_text: str, max_results: int) -> list[dict[str, str]]:
    """解析 Google 新闻中文区 RSS，作为中文搜索的稳定降级来源。"""
    root = ElementTree.fromstring(xml_text)
    parsed_results: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        if not title or not url:
            continue

        published_at = (item.findtext("pubDate") or "").strip()
        try:
            published_at = parsedate_to_datetime(published_at).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OverflowError):
            published_at = published_at or "日期未知"

        description = BeautifulSoup(
            item.findtext("description") or "",
            "html.parser",
        ).get_text(" ", strip=True)
        parsed_results.append({
            "title": title,
            "href": url,
            "body": description or "未提供摘要",
            "date": published_at,
        })
        if len(parsed_results) >= max_results:
            break
    return parsed_results


def search_360_chinese_web(
    query: str,
    max_results: int,
    recent_only: bool,
) -> list[dict[str, str]]:
    """通过面向中国互联网的 360 搜索执行一次中文网页检索。"""
    current_year = datetime.now().year
    effective_query = query.strip()
    if recent_only and not re.search(r"20\d{2}", effective_query):
        effective_query = f"{effective_query} {current_year} 最新"

    response = httpx.get(
        "https://m.so.com/s",
        params={"q": effective_query},
        headers=CHINESE_SEARCH_HEADERS,
        follow_redirects=True,
        timeout=25,
    )
    response.raise_for_status()
    results = parse_360_search_results(response.text, max_results)
    if not results:
        raise RuntimeError("360 搜索未返回可解析的中文网页结果")
    return results


def search_google_news_chinese(
    query: str,
    max_results: int,
    recent_only: bool,
) -> list[dict[str, str]]:
    """搜索 Google 新闻中文区，补充中文媒体中的近期公开信息。"""
    current_year = datetime.now().year
    effective_query = query.strip()
    if recent_only and not re.search(r"20\d{2}", effective_query):
        effective_query = f"{effective_query} {current_year} 最新"
    response = httpx.get(
        "https://news.google.com/rss/search",
        params={
            "q": effective_query,
            "hl": "zh-CN",
            "gl": "CN",
            "ceid": "CN:zh-Hans",
        },
        headers=CHINESE_SEARCH_HEADERS,
        follow_redirects=True,
        timeout=25,
    )
    response.raise_for_status()
    results = parse_google_news_results(response.text, max_results)
    if not results:
        raise RuntimeError("Google 新闻中文区未返回可解析结果")
    return results


def run_chinese_web_search(
    query: str,
    max_results: int,
    recent_only: bool,
) -> list[dict[str, str]]:
    """同步执行中文网页搜索；移动 360 为主通道，Google 中文新闻为降级通道。"""
    errors: list[str] = []

    # 五个 Researcher 会并行运行，但公共搜索站点不适合瞬时承受十几条并发请求。
    # 在工具层限制为 2 路，既保留 Agent 并行，也减少限流、验证码和超时。
    with CHINESE_SEARCH_CONCURRENCY:
        try:
            return search_360_chinese_web(query, max_results, recent_only)
        except Exception as exc:
            errors.append(f"360 中文搜索失败：{exc}")

        try:
            return search_google_news_chinese(query, max_results, recent_only)
        except Exception as exc:
            errors.append(f"Google 新闻中文区失败：{exc}")

    raise RuntimeError("；".join(errors))


def format_chinese_web_search_results(
    query: str,
    results: list[dict[str, str]],
) -> str:
    """把中文网页搜索结果转换为 Researcher 可引用的文本。"""
    lines = [f"查询：{query}"]
    if not results:
        lines.append("未找到中文网页结果，请调整中文关键词后重试。")
    for index, result in enumerate(results, start=1):
        lines.append(
            f"[{index}] {result.get('title') or '无标题'}\n"
            f"URL：{result.get('href') or result.get('url') or ''}\n"
            f"发布日期：{result.get('date') or '日期未知'}\n"
            f"摘要：{result.get('body') or result.get('snippet') or '未提供摘要'}"
        )
    return "\n\n".join(lines)


@tool(
    "web_search",
    description=(
        "使用面向中国市场的中文网页搜索检索公开互联网，无需 API Key。"
        "查询词应使用中文；研究近期动态时应在查询中包含当前年份或明确时间范围，并将 recent_only 设为 true。"
    ),
)
async def chinese_web_search(
    queries: List[str],
    recent_only: bool = False,
    max_results: Annotated[int, InjectedToolArg] = 5,
) -> str:
    """执行中文区域 Web 搜索，并返回标题、直达 URL、发布日期和摘要。"""
    async def search_one(query: str) -> str:
        try:
            results = await asyncio.to_thread(
                run_chinese_web_search,
                query,
                max_results,
                recent_only,
            )
            return format_chinese_web_search_results(query, results)
        except Exception as exc:
            return f"查询：{query}\n搜索失败：{exc}"

    valid_queries = [query.strip() for query in queries if query and query.strip()]
    if not valid_queries:
        return "未收到有效的中文搜索关键词。"

    # 每个维度先执行一条高覆盖综合查询。需要补证时，Researcher 可在下一轮再次调用工具；
    # 这样能避免五个并行 Agent 一次发出十几条请求，触发公共搜索站点的验证码。
    section = await search_one(valid_queries[0])
    if len(valid_queries) > 1:
        section += "\n\n提示：为保证公共搜索稳定性，本轮仅执行第一条综合查询；如需补证，请下一轮再次调用 web_search。"
    return section

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """异步执行多个 Tavily 搜索查询。

    参数：
        search_queries: 需要执行的搜索查询字符串列表
        max_results: 每个查询最多返回的结果数
        topic: 用于过滤结果的主题类别
        include_raw_content: 是否包含完整网页内容
        config: 用于读取 API Key 的运行配置
        
    返回：
        Tavily API 返回的搜索结果字典列表
    """
    # 使用配置中的 API Key 初始化 Tavily Client
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # 创建需要并行执行的搜索任务
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # 并行执行全部搜索查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型总结网页内容，并提供超时保护。
    
    参数：
        model: 配置为执行总结任务的 Chat Model
        webpage_content: 需要总结的网页原始内容
        
    返回：
        包含关键摘录的格式化摘要；如果总结失败，则返回原始内容
    """
    try:
        # 创建包含当前日期上下文的 Prompt
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # 执行总结并设置超时，避免任务卡死
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 总结任务的超时时间为 60 秒
        )
        
        # 使用结构化章节格式化摘要
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # 总结超时时返回原始内容
        logging.warning("网页总结在 60 秒后超时，将返回原始内容")
        return webpage_content
    except Exception as e:
        # 其他总结异常：记录日志并返回原始内容
        logging.warning(f"网页总结失败：{str(e)}，将返回原始内容")
        return webpage_content

##########################
# 反思工具
##########################

@tool(description="用于研究规划的策略反思工具")
def think_tool(reflection: str) -> str:
    """用于反思研究进展和辅助决策的策略工具。

    每次搜索后使用此工具分析结果并系统规划下一步，
    从而在研究工作流中主动暂停并进行质量判断。

    使用时机：
    - 收到搜索结果后：找到了哪些关键信息？
    - 决定下一步之前：现有信息能否完整回答问题？
    - 评估研究缺口时：还缺少哪些具体信息？
    - 结束研究之前：现在能否给出完整答案？

    反思应覆盖：
    1. 当前发现分析——已经收集到哪些具体信息？
    2. 缺口评估——还缺少哪些关键信息？
    3. 质量评估——是否有足够的证据或示例支撑优质答案？
    4. 策略决策——应该继续搜索，还是提交答案？

    参数：
        reflection: 对研究进展、发现、缺口和下一步的详细反思

    返回：
        确认已经记录反思内容，供后续决策使用
    """
    return f"已记录研究反思：{reflection}"

##########################
# MCP 工具
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """通过 OAuth Token Exchange 将 Supabase Token 换取为 MCP Access Token。
    
    参数：
        supabase_token: 有效的 Supabase 身份验证 Token
        base_mcp_url: MCP 服务器的 Base URL
        
    返回：
        成功时返回 Token 数据字典，失败时返回 None
    """
    try:
        # 准备 OAuth Token Exchange 请求数据
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # 执行 Token Exchange 请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获得 Token
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误详情以便调试
                    response_text = await response.text()
                    logging.error(f"Token Exchange 失败：{response_text}")
                    
    except Exception as e:
        logging.error(f"Token Exchange 过程中发生错误：{e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """读取已保存的身份验证 Token，并验证是否过期。
    
    参数：
        config: 包含 Thread 和用户标识的运行配置
        
    返回：
        Token 有效且未过期时返回 Token 字典，否则返回 None
    """
    store = get_store()
    
    # 从配置中提取必要标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # 读取已保存的 Token
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # 检查 Token 是否过期
    expires_in = tokens.value.get("expires_in")  # 距离过期的秒数
    created_at = tokens.created_at  # Token 创建时间
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token 已过期，清理后返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """在配置存储中保存身份验证 Token。
    
    参数：
        config: 包含 Thread 和用户标识的运行配置
        tokens: 需要保存的 Token 字典
    """
    store = get_store()
    
    # 从配置中提取必要标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # 保存 Token
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP Token，必要时申请新的 Token。
    
    参数：
        config: 包含身份验证信息的运行配置
        
    返回：
        有效的 Token 字典；无法获取时返回 None
    """
    # 优先尝试读取已有的有效 Token
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # 提取 Supabase Token，用于申请新的 Token
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # 使用 Supabase Token 换取 MCP Token
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 保存并返回新 Token
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """为 MCP 工具封装完整的身份验证和异常处理。
    
    参数：
        tool: 需要封装的 MCP StructuredTool
        
    返回：
        增加了身份验证异常处理的工具
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """增加 MCP 异常处理和易读提示的协程。"""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在异常链中递归查找 MCP 异常。"""
            if isinstance(exc, McpError):
                return exc
            
            # 通过属性检查处理 ExceptionGroup（Python 3.11 及以上）
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # 执行工具原始功能
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # 在异常链中查找 MCP 特定异常
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 非 MCP 异常，重新抛出原始异常
                raise original_error
            
            # 处理 MCP 特定异常
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # 检查需要身份验证或用户交互的错误
            if error_code == -32003:  # 需要交互的错误码
                message_payload = error_data.get("message", {})
                error_message = "需要用户完成交互"
                
                # 尽可能提取用户易读的消息
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # 如果存在 URL，则附加到消息中供用户访问
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # 其他 MCP 异常直接重新抛出
            raise original_error
    
    # 使用增强版协程替换工具原协程
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置支持身份验证的 MCP（Model Context Protocol）工具。
    
    参数：
        config: 包含 MCP 服务器详情的运行配置
        existing_tool_names: 已经使用的工具名称集合，用于避免冲突
        
    返回：
        已配置并可直接使用的 MCP 工具列表
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 步骤 1：根据需要处理身份验证
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # 步骤 2：验证必要配置
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # 步骤 3：建立 MCP 服务器连接
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # 存在 Token 时配置身份验证 Header
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO：OAP 合并多 MCP Server 支持后，需要更新此处代码
    
    # 步骤 4：从 MCP 服务器加载工具
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # MCP 服务器连接失败时返回空列表
        return []
    
    # 步骤 5：过滤并配置工具
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳过名称冲突的工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP 工具 '{mcp_tool.name}' 与已有工具重名，已跳过"
            )
            continue
        
        # 只加载配置中明确指定的工具
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # 为工具封装身份验证处理后加入列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# 通用工具
##########################

async def get_search_tool(search_api: SearchAPI):
    """根据指定的 API 提供商配置并返回搜索工具。
    
    参数：
        search_api: 需要使用的 Search API 提供商
        
    返回：
        为指定提供商配置完成的搜索工具对象列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # 带调用次数限制的 Anthropic 原生 Web Search
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI 的 Web Search Preview 功能
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # 为 Tavily 搜索工具配置 metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.BING:
        # 保留 BING 配置值兼容已有环境，但执行逻辑已从 RSS 替换为中文区域 Web 搜索。
        return [chinese_web_search]
        
    elif search_api == SearchAPI.NONE:
        # 未配置搜索功能
        return []
        
    # 未知 Search API 类型的默认降级处理
    return []
    
async def get_all_tools(config: RunnableConfig):
    """组装包含研究、搜索和 MCP 工具的完整工具集。
    
    参数：
        config: 指定 Search API 和 MCP 设置的运行配置
        
    返回：
        完成配置并可用于研究的全部工具列表
    """
    # 先加入核心研究工具
    tools = [tool(ResearchComplete), think_tool]
    
    # 加入配置的搜索工具
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # 记录已有工具名称，避免发生冲突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # 如果配置了 MCP，则加入对应工具
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具调用消息中提取研究笔记。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型提供商原生 Web Search 工具
##########################

def anthropic_websearch_called(response):
    """检测响应中是否使用了 Anthropic 原生 Web Search。
    
    参数：
        response: Anthropic API 返回的响应对象
        
    返回：
        调用过 Web Search 时返回 True，否则返回 False
    """
    try:
        # 读取响应 metadata 结构
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # 检查服务端工具使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # 查找 Web Search 请求次数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # 发生过 Web Search 请求时返回 True
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # 处理响应结构不符合预期的情况
        return False

def openai_websearch_called(response):
    """检测响应中是否使用了 OpenAI Web Search。
    
    参数：
        response: OpenAI API 返回的响应对象
        
    返回：
        调用过 Web Search 时返回 True，否则返回 False
    """
    # 检查响应 metadata 中的工具输出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # 在工具输出中查找 Web Search 调用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token 超限检测工具
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断异常是否表示 Token 或上下文长度超过限制。
    
    参数：
        exception: 需要分析的异常
        model_name: 可选的模型名称，用于提高提供商识别准确度
        
    返回：
        异常表示 Token 超限时返回 True，否则返回 False
    """
    error_str = str(exception).lower()
    
    # 步骤 1：尽可能根据模型名称判断提供商
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # 步骤 2：检查对应提供商的 Token 超限特征
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # 步骤 3：提供商未知时检查全部提供商
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 OpenAI Token 超限。"""
    # 分析异常 metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 检查是否为 OpenAI 异常
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # 检查常见的 OpenAI Token 超限异常类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # 在错误消息中查找 Token 相关关键词
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # 检查特定的 OpenAI 错误码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Anthropic Token 超限。"""
    # 分析异常 metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 检查是否为 Anthropic 异常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # 检查 Anthropic 特有的错误特征
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic 使用特定错误消息表示 Token 超限
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Google/Gemini Token 超限。"""
    # 分析异常 metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 检查是否为 Google/Gemini 异常
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # 检查 Google 特有的资源耗尽异常
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # 检查 Google API 特有的资源耗尽特征
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# 注意：此列表可能过时或不适用于当前模型，请根据实际情况更新
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查询指定模型的 Token 上限。
    
    参数：
        model_string: 需要查询的模型标识字符串
        
    返回：
        找到时返回整数形式的 Token 上限；模型不在映射表中时返回 None
    """
    # 查询已知模型的 Token 上限
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # 映射表中未找到该模型
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """删除最后一条 AI 消息及其后的内容，以截断消息历史。
    
    此方法通过删除最近的上下文来处理 Token 超限异常。
    
    参数：
        messages: 需要截断的消息对象列表
        
    返回：
        截断到最后一条 AI 消息之前的消息列表，不包含该 AI 消息
    """
    # 从后向前查找最后一条 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回最后一条 AI 消息之前的全部内容，不包含该消息
            return messages[:i]
    
    # 没有找到 AI 消息时返回原列表
    return messages

##########################
# 其他辅助方法
##########################

def get_today_str() -> str:
    """获取用于 Prompt 和输出展示的格式化当前日期。
    
    返回：
        易读的日期字符串，例如 `Mon Jan 15, 2024`
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置中提取值，并处理 Enum 和 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境变量或配置中获取指定模型的 API Key。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        elif model_name.startswith("deepseek:"):
            return api_keys.get("DEEPSEEK_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        elif model_name.startswith("deepseek:"):
            return os.getenv("DEEPSEEK_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """从环境变量或配置中获取 Tavily API Key。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
