"""Open Deep Research 系统的配置管理。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用 Search API 提供商的枚举。"""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    BING = "bing"
    NONE = "none"

class MCPConfig(BaseModel):
    """Model Context Protocol（MCP）服务器配置。"""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """MCP 服务器 URL。"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """向 LLM 开放的工具。"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """MCP 服务器是否要求身份验证。"""

class Configuration(BaseModel):
    """Deep Research Agent 的主配置类。"""
    
    # 通用配置
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "模型进行结构化输出调用时的最大重试次数"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否允许 Researcher 在开始研究前向用户提出澄清问题"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 5,
                "max": 20,
                "step": 1,
                "description": "允许同时运行的最大研究单元数。Researcher 可以借此使用多个子 Agent 开展研究；并发数越高，越可能触发 API 限流。"
            }
        }
    )
    # 研究配置
    search_api: SearchAPI = Field(
        default=SearchAPI.BING,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "bing",
                "description": "研究使用的 Search API。请确认 Researcher Model 支持所选 Search API。",
                "options": [
                    {"label": "中文网页搜索（中国区、免费、无需 Key）", "value": SearchAPI.BING.value},
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI 原生 Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic 原生 Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "不使用搜索", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Research Supervisor 的最大研究轮数，即 Supervisor 反思研究结果并提出后续问题的最大次数。"
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "单个 Researcher 执行过程中允许的最大工具调用轮数。"
            }
        }
    )
    # 模型配置
    summarization_model: str = Field(
        default="deepseek:deepseek-v4-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-v4-flash",
                "description": "用于总结 Tavily 搜索结果的模型"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "总结模型的最大输出 Token 数"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "网页内容进入总结流程前允许的最大字符数"
            }
        }
    )
    research_model: str = Field(
        default="deepseek:deepseek-v4-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-v4-flash",
                "description": "用于执行研究的模型。请确认 Researcher Model 支持所选 Search API。"
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "研究模型的最大输出 Token 数"
            }
        }
    )
    compression_model: str = Field(
        default="deepseek:deepseek-v4-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-v4-flash",
                "description": "用于压缩子 Agent 研究发现的模型。请确认 Compression Model 支持所选 Search API。"
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "压缩模型的最大输出 Token 数"
            }
        }
    )
    final_report_model: str = Field(
        default="deepseek:deepseek-v4-pro",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-v4-pro",
                "description": "根据全部研究发现撰写最终报告的模型"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "最终报告模型的最大输出 Token 数"
            }
        }
    )
    # MCP 服务器配置
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP 服务器配置"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "传递给 Agent 的补充指令，用于说明可用的 MCP 工具。"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """根据 RunnableConfig 创建 Configuration 实例。"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic 配置。"""
        
        arbitrary_types_allowed = True
