# CompeteX：AI 竞品情报研究 Agent

CompeteX 是基于 LangChain 官方 Open Deep Research 二次设计的个人产品原型。它把一次竞品研究拆成需求澄清、研究规划、并行检索、证据压缩、覆盖度反思和报告生成，并要求最终报告区分已验证事实、分析判断和产品建议。

## 产品定位

目标用户是需要持续追踪市场的 AI 产品经理、战略分析师和创业团队。它不是“替人做决定”的聊天机器人，而是帮助用户更快形成有来源、可复核的决策材料。

核心价值：

- 把跨官网、文档、新闻和社区的手工搜索变成有状态的研究流程；
- 多个研究单元可以并行工作，并在信息不足时补充搜索；
- 强制保留引用，对重要结论标注证据置信度；
- 通过并发数、研究轮数和工具调用次数控制延迟与成本。

## 当前演示版改造

- LangGraph Studio 中的图已命名为 `CompeteX Research Agent`；
- 移除仅适用于线上部署的 Supabase 鉴权，方便本地面试演示；
- 默认并行研究单元从 5 降为 3，研究轮数从 6 降为 3，单研究员工具调用上限从 10 降为 5；
- 竞品类任务默认输出决策摘要、定位、用户、能力、商业模式、近期信号、风险、机会、证据置信度和待验证事项；
- 重要事实优先引用官网、定价页、产品文档、更新日志和官方仓库。

这些默认值面向低成本原型演示，不代表生产环境最佳配置。

## 运行条件

已安装 Python 虚拟环境与全部依赖。默认模型已切换为 DeepSeek：研究与压缩使用 `deepseek-v4-flash`，最终报告使用 `deepseek-v4-pro`。网页搜索使用免费的中文 Web 搜索（移动 360 主通道、Google 中文新闻降级），因此只需要一个 `DEEPSEEK_API_KEY`。

将密钥填入项目根目录 `.env`。不要将 `.env` 提交到 Git。

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
```

## 一键启动

推荐使用中文面试演示界面。在 PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'E:\AI项目\CompeteX-LangGraph'
.\scripts\start_interview_ui.ps1
```

脚本会同时启动 LangGraph API 与中文 UI，并自动打开：

- 中文研究台：<http://127.0.0.1:3000>
- API 文档：<http://127.0.0.1:2030/docs>

按 `Ctrl+C` 可同时停止两个服务。

## Vercel 在线部署

仓库根目录已包含 Vercel 所需的 FastAPI 入口 `app.py` 和 `vercel.json`。导入 GitHub 仓库即可同时部署静态研究台与 LangGraph API，无需在 Vercel 中保存模型密钥。

在线版采用 BYOK（自带密钥）：用户在“配置 Key”窗口选择模型服务商并填写 API Key。密钥只保存在当前页面内存中，随单次 HTTPS 请求发送到本项目的 Vercel 函数，再由函数转发给模型服务商；刷新页面后自动清除。中文 Web 搜索默认免费且无需 Key，也可切换为 Tavily 并手动填写 Tavily Key。

Vercel 函数最长运行时间设置为 300 秒。研究范围过大时，建议减少对比对象或缩小研究维度后重试。

如需只启动 LangGraph Studio，在 PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'E:\AI项目\CompeteX-LangGraph'
.\scripts\start_demo.ps1
```

随后打开：

- Studio：<https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>
- API 文档：<http://127.0.0.1:2024/docs>
- 健康检查：<http://127.0.0.1:2024/ok>

首次演示前建议登录 LangSmith。启动脚本只绑定 `127.0.0.1`，不会向局域网暴露服务。

## 推荐演示任务

将下面内容作为用户消息：

> 请为计划进入中国市场的 AI 会议助手团队，对比 Otter.ai、Fireflies.ai 和 Notion AI Meeting Notes。研究范围限定为截至今天可公开验证的信息，重点覆盖目标用户、核心工作流、定价、集成生态、差异化、近期产品动态和用户采用阻力。优先引用官网、定价页、帮助文档和更新日志。区分事实、推断与建议；信息不足时明确说明。最后给出一个面向 20—200 人中国科技公司的 MVP 机会清单，并按价值、可行性和风险排序。

演示时重点展示：

1. Agent 是否先确认研究边界；
2. Supervisor 如何把任务拆成多个并行研究单元；
3. 研究过程中是否出现“信息是否充分”的反思循环；
4. 最终报告中的引用、置信度和待验证事项；
5. Studio 中每个节点的输入、输出、耗时和调用轨迹。

更多题目见 [demo/DEMO_QUESTIONS_CN.md](demo/DEMO_QUESTIONS_CN.md)。

## 面试现场的诚实表述

推荐说法：

> 这是我基于 LangChain 官方开源项目完成的个人复现与垂直场景产品化设计。我没有重新发明底层研究框架，主要工作是定义用户场景、报告标准、证据策略、成本边界、评测方案和演示流程，并完成本地部署验证。

不要把官方基准成绩说成个人项目成绩，也不要把测试数据说成线上业务数据。

## 已知边界

- 当前本地演示需要外部模型和搜索密钥；
- 搜索结果可能受网页可访问性和时效性影响；
- 引用存在不等于结论一定正确，关键结论仍需人工复核；
- 当前是开发服务器和内存状态，不用于生产部署；
- 为方便面试演示移除了本地鉴权，若上线必须重新接入鉴权、权限隔离和审计。

## 文档导航

- [完整面试故事线](INTERVIEW_STORY_CN.md)
- [产品方案与评测设计](docs/PRODUCT_DESIGN_CN.md)
- [演示题库](demo/DEMO_QUESTIONS_CN.md)
- [原项目 README](README.md)
