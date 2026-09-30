"""CompeteX 五维研究与中文网页搜索的回归测试。"""

import asyncio

from langchain_core.messages import AIMessage

import open_deep_research.deep_researcher as research_module
from open_deep_research.deep_researcher import (
    REQUIRED_COMPETITOR_DIMENSIONS,
    build_required_research_tool_calls,
    format_research_failure,
    format_structured_research_result,
)
from open_deep_research.state import ResearchEvidence, StructuredResearchResult
from open_deep_research.utils import (
    chinese_web_search,
    format_chinese_web_search_results,
    parse_360_search_results,
    parse_google_news_results,
)


def test_required_research_calls_cover_five_dimensions() -> None:
    calls = build_required_research_tool_calls("对比豆包和 DeepSeek")

    assert len(calls) == 5
    assert tuple(call["args"]["research_dimension"] for call in calls) == REQUIRED_COMPETITOR_DIMENSIONS
    assert len({call["id"] for call in calls}) == 5
    assert all(call["name"] == "ConductResearch" for call in calls)
    assert all("对比豆包和 DeepSeek" in call["args"]["research_topic"] for call in calls)


def test_supervisor_executes_five_required_researchers_concurrently(monkeypatch) -> None:
    active_count = 0
    peak_count = 0

    class StubResearcherSubgraph:
        async def ainvoke(self, state, config):
            nonlocal active_count, peak_count
            active_count += 1
            peak_count = max(peak_count, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            return {
                "compressed_research": f"## 【{state['research_dimension']}】\n测试结果",
                "raw_notes": [state["research_dimension"]],
            }

    monkeypatch.setattr(research_module, "researcher_subgraph", StubResearcherSubgraph())
    calls = build_required_research_tool_calls("对比豆包和 DeepSeek")
    state = {
        "supervisor_messages": [AIMessage(content="", tool_calls=calls)],
        "research_brief": "对比豆包和 DeepSeek",
        "research_iterations": 1,
    }

    command = asyncio.run(research_module.supervisor_tools(
        state,
        {"configurable": {"max_concurrent_research_units": 2}},
    ))

    assert peak_count == 5
    assert command.goto == "supervisor"
    assert len(command.update["supervisor_messages"]) == 5


def test_one_researcher_failure_does_not_discard_other_results(monkeypatch) -> None:
    class PartiallyFailingSubgraph:
        async def ainvoke(self, state, config):
            if state["research_dimension"] == "近期动态":
                raise RuntimeError("模拟单分支限流")
            return {
                "compressed_research": f"## 【{state['research_dimension']}】\n测试结果",
                "raw_notes": [state["research_dimension"]],
            }

    monkeypatch.setattr(research_module, "researcher_subgraph", PartiallyFailingSubgraph())
    calls = build_required_research_tool_calls("对比豆包和 DeepSeek")
    state = {
        "supervisor_messages": [AIMessage(content="", tool_calls=calls)],
        "research_brief": "对比豆包和 DeepSeek",
        "research_iterations": 1,
    }

    command = asyncio.run(research_module.supervisor_tools(state, {"configurable": {}}))
    messages = command.update["supervisor_messages"]

    assert command.goto == "supervisor"
    assert len(messages) == 5
    assert sum("测试结果" in message.content for message in messages) == 4
    assert "模拟单分支限流" in messages[-1].content


def test_structured_research_result_has_stable_markdown_sections() -> None:
    result = StructuredResearchResult(
        research_dimension="产品定位",
        summary="两款产品定位不同。",
        key_findings=["豆包更偏大众助手。"],
        comparison_points=["两者目标场景不同。"],
        evidence=[
            ResearchEvidence(
                source_title="官方说明",
                url="https://example.com/product",
                published_at="2026-09-01",
                evidence="官方页面说明了产品定位。",
                confidence="高",
            )
        ],
        evidence_gaps=["缺少统一口径的活跃用户数据。"],
    )

    output = format_structured_research_result(result, "产品定位")

    for heading in ("## 【产品定位】", "### 结论摘要", "### 关键发现", "### 横向对比", "### 结构化证据", "### 证据缺口", "### 来源"):
        assert heading in output
    assert "https://example.com/product" in output


def test_research_failure_is_returned_as_structured_evidence_gap() -> None:
    output = format_research_failure("近期动态", RuntimeError("模拟限流"))

    assert "## 【近期动态】" in output
    assert "### 证据缺口" in output
    assert "RuntimeError：模拟限流" in output


def test_chinese_web_search_formats_results() -> None:
    assert chinese_web_search.name == "web_search"
    assert "中文网页搜索" in chinese_web_search.description
    output = format_chinese_web_search_results(
        "豆包最新动态",
        [{
            "title": "中文产品最新动态",
            "href": "https://example.cn/news",
            "body": "这里是中文网页摘要。",
            "date": "2026年9月1日",
        }],
    )
    assert "查询：豆包最新动态" in output
    assert "中文产品最新动态" in output
    assert "https://example.cn/news" in output
    assert "2026年9月1日" in output


def test_360_search_parser_returns_direct_source_url() -> None:
    html = """
    <ul class="result">
      <li class="res-list">
        <h3 class="res-title">
          <a href="https://www.so.com/link?m=redirect"
             data-mdurl="https://example.cn/direct-news">豆包最新发布</a>
        </h3>
        <div class="res-rich">
          <span class="g-c-gray">2026年9月1日 - </span>
          <span class="res-list-summary">这里是中文网页摘要。</span>
        </div>
      </li>
    </ul>
    """

    results = parse_360_search_results(html, max_results=5)

    assert results == [{
        "title": "豆包最新发布",
        "href": "https://example.cn/direct-news",
        "body": "这里是中文网页摘要。",
        "date": "2026年9月1日",
    }]


def test_mobile_360_search_parser_returns_direct_source_url() -> None:
    html = """
    <div class="g-card res-list" data-pcurl="https://example.cn/mobile-direct">
      <a class="alink" href="https://m.so.com/jump?u=redirect">
        <h3 class="res-title">豆包与 DeepSeek 对比</h3>
        <p class="g-main summary">移动中文搜索摘要。</p>
      </a>
      <div class="res-supplement"><time>9月14日</time></div>
    </div>
    """

    results = parse_360_search_results(html, max_results=5)

    assert results[0]["href"] == "https://example.cn/mobile-direct"
    assert results[0]["body"] == "移动中文搜索摘要。"


def test_google_news_parser_keeps_date_and_link() -> None:
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel><item>
      <title>DeepSeek 最新动态</title>
      <link>https://news.google.com/rss/articles/example</link>
      <pubDate>Mon, 14 Sep 2026 08:00:00 GMT</pubDate>
      <description><![CDATA[<a href="https://example.cn">中文新闻摘要</a>]]></description>
    </item></channel></rss>
    """

    results = parse_google_news_results(xml_text, max_results=5)

    assert results[0]["href"] == "https://news.google.com/rss/articles/example"
    assert results[0]["date"] == "2026-09-14"
    assert "中文新闻摘要" in results[0]["body"]
