const API_BASE = window.location.origin;

const examples = [
  { title: "AI 会议助手", meta: "市场进入与 MVP 机会", prompt: "请为计划进入中国市场的 AI 会议助手团队，对比 Otter.ai、Fireflies.ai 和 Notion AI Meeting Notes。重点覆盖目标用户、核心工作流、定价、集成生态、差异化、近期产品动态和用户采用阻力。区分事实、推断与建议，最后给出面向中国科技公司的 MVP 机会清单。" },
  { title: "AI 编程工具", meta: "产品定位与差异化", prompt: "对比 Cursor、GitHub Copilot 与 Windsurf 的目标用户、核心功能、定价和差异化，分析中国开发者工具团队可切入的机会。重要事实请保留来源。" },
  { title: "智能客服平台", meta: "企业采购决策", prompt: "以一家 200 人电商企业的采购视角，对比三类主流 AI 智能客服方案，重点分析部署成本、知识库效果、人工接管、安全合规与实施风险。" }
];

const providerDefaults = {
  deepseek: { label: "DeepSeek API Key", key: "deepseek", research: "deepseek:deepseek-chat", report: "deepseek:deepseek-chat", placeholder: "sk-..." },
  openai: { label: "OpenAI API Key", key: "openai", research: "openai:gpt-4.1-mini", report: "openai:gpt-4.1", placeholder: "sk-..." },
  anthropic: { label: "Anthropic API Key", key: "anthropic", research: "anthropic:claude-sonnet-4-5", report: "anthropic:claude-sonnet-4-5", placeholder: "sk-ant-..." },
  google: { label: "Google API Key", key: "google", research: "google_genai:gemini-2.5-flash", report: "google_genai:gemini-2.5-pro", placeholder: "AIza..." }
};

const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
let running = false;
let timer = null;
let startTime = 0;
let runtimeConfig = null;
let conversationHistory = [];
let completedNodes = new Set();

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function renderMarkdown(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  safe = safe.replace(/^### (.+)$/gm, "<h3>$1</h3>").replace(/^## (.+)$/gm, "<h2>$1</h2>").replace(/^# (.+)$/gm, "<h1>$1</h1>");
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  safe = safe.replace(/^[-*] (.+)$/gm, "<li>$1</li>").replace(/(?:<li>.*<\/li>\n?)+/g, (items) => `<ul>${items}</ul>`);
  return safe.split(/\n{2,}/).map((block) => /^<(h\d|ul)/.test(block) ? block : `<p>${block.replace(/\n/g, "<br>")}</p>`).join("");
}

function addMessage(role, content, loading = false) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  item.innerHTML = `<div class="avatar">${role === "user" ? "我" : "CX"}</div><div class="bubble">${loading ? '<div class="thinking"><i class="spinner"></i><span>正在理解需求并规划研究路径…</span></div>' : renderMarkdown(content)}</div>`;
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item.querySelector(".bubble");
}

function resetRun() {
  completedNodes.clear();
  clearInterval(timer);
  $("#elapsed").textContent = "00:00";
  $("#sourceCount").textContent = "0";
  $("#runState").className = "run-state";
  $("#runState").textContent = "待开始";
  $("#taskBadge").textContent = "待开始";
  document.querySelectorAll(".timeline li").forEach((item) => {
    item.className = "";
    item.querySelector("time").textContent = "待处理";
  });
  updateProgress();
}

function updateProgress() {
  const count = completedNodes.size;
  const percent = Math.round(count / 4 * 100);
  $("#progressBar").style.width = `${percent}%`;
  $("#progressText").textContent = `${percent}%`;
  $("#nodeCount").textContent = `${count}/4`;
}

function markNode(name, done = false) {
  const aliases = {
    clarify_with_user: "clarify_with_user",
    write_research_brief: "write_research_brief",
    research_supervisor: "research_supervisor",
    supervisor: "research_supervisor",
    researcher: "research_supervisor",
    compress_research: "research_supervisor",
    final_report_generation: "final_report_generation"
  };
  const match = Object.keys(aliases).find((key) => name.includes(key));
  if (!match) return;
  const key = aliases[match];
  const item = document.querySelector(`[data-node="${key}"]`);
  if (done) {
    item.className = "done";
    item.querySelector("time").textContent = "已完成";
    completedNodes.add(key);
  } else {
    item.className = "active";
    item.querySelector("time").textContent = "执行中";
  }
  updateProgress();
}

function friendlyErrorMessage(error) {
  const message = String(error?.message || error || "未知错误");
  if (/401|authentication|api.?key|unauthorized/i.test(message)) return "模型服务拒绝了请求，请检查 API Key 和所选模型是否匹配。";
  if (/timeout|timed out|FUNCTION_INVOCATION_TIMEOUT/i.test(message)) return "本次研究超过了云函数运行时长。请缩小研究范围后重试。";
  if (/Failed to fetch|NetworkError/i.test(message)) return "无法连接 CompeteX 研究服务，请检查网络后重试。";
  return message;
}

function updateProviderDefaults() {
  const defaults = providerDefaults[$("#provider").value];
  $("#modelKeyLabel").textContent = defaults.label;
  $("#modelKey").placeholder = defaults.placeholder;
  $("#researchModel").value = defaults.research;
  $("#reportModel").value = defaults.report;
}

function updateSearchFields() {
  const requiresTavily = $("#searchApi").value === "tavily";
  $("#tavilyField").hidden = !requiresTavily;
  $("#tavilyKey").required = requiresTavily;
}

function openConfig() {
  if (!$("#configDialog").open) $("#configDialog").showModal();
  window.setTimeout(() => $("#modelKey").focus(), 0);
}

function saveConfig() {
  const provider = $("#provider").value;
  const defaults = providerDefaults[provider];
  const keys = { deepseek: null, openai: null, anthropic: null, google: null, tavily: null };
  keys[defaults.key] = $("#modelKey").value.trim();
  keys.tavily = $("#tavilyKey").value.trim() || null;
  runtimeConfig = {
    keys,
    search_api: $("#searchApi").value,
    research_model: $("#researchModel").value.trim(),
    final_report_model: $("#reportModel").value.trim(),
    allow_clarification: $("#allowClarification").checked
  };
  $("#openConfig span").textContent = "配置已就绪";
  $("#openConfig").classList.add("ready");
  $("#configDialog").close();
}

async function checkConnection() {
  const element = $("#connection");
  try {
    const response = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    if (!response.ok) throw new Error();
    element.className = "connection online";
    element.querySelector("span").textContent = "研究服务已连接";
  } catch {
    element.className = "connection offline";
    element.querySelector("span").textContent = "研究服务不可用";
  }
}

async function consumeStream(response, answer) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalContent = "";
  let outcome = "awaiting_input";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "progress") markNode(event.node, true);
      if (["message", "result", "complete"].includes(event.type) && event.content) {
        finalContent = event.content;
        answer.innerHTML = renderMarkdown(event.content);
        $("#sourceCount").textContent = String((event.content.match(/https?:\/\//g) || []).length);
      }
      if (event.type === "complete") outcome = event.status;
    }
  }
  return { content: finalContent, outcome };
}

async function submitResearch(prompt) {
  if (running) return;
  if (!runtimeConfig) {
    openConfig();
    return;
  }

  running = true;
  resetRun();
  $("#sendButton").disabled = true;
  $("#sendButton span").textContent = "研究进行中";
  $("#runState").className = "run-state running";
  $("#runState").textContent = "运行中";
  $("#taskBadge").textContent = "进行中";
  document.querySelector(".welcome-card")?.remove();
  addMessage("user", prompt);
  conversationHistory.push({ role: "user", content: prompt });
  const answer = addMessage("assistant", "", true);
  startTime = Date.now();
  timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - startTime) / 1000);
    $("#elapsed").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }, 1000);

  try {
    const response = await fetch(`${API_BASE}/api/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversationHistory, ...runtimeConfig })
    });
    if (!response.ok) throw new Error(`研究请求失败（${response.status}）：${await response.text()}`);
    answer.innerHTML = '<div class="thinking"><i class="spinner"></i><span>研究节点正在运行，结果将实时呈现…</span></div>';
    const result = await consumeStream(response, answer);
    conversationHistory.push({ role: "assistant", content: result.content });

    if (result.outcome === "complete") {
      ["clarify_with_user", "write_research_brief", "research_supervisor", "final_report_generation"].forEach((node) => markNode(node, true));
      $("#runState").className = "run-state done";
      $("#runState").textContent = "已完成";
      $("#taskBadge").textContent = "已完成";
      $("#interviewTip").textContent = "最终建议由多轮检索证据和覆盖度反思逐步收敛而来；关键结论仍建议人工复核。";
    } else {
      $("#runState").className = "run-state running";
      $("#runState").textContent = "等待补充";
      $("#taskBadge").textContent = "待补充";
      $("#interviewTip").textContent = "Agent 正在澄清研究范围。回答它的问题后，将在同一上下文中继续研究。";
    }
  } catch (error) {
    answer.innerHTML = `<p><strong>运行失败</strong></p><p>${escapeHtml(friendlyErrorMessage(error))}</p>`;
    $("#runState").className = "run-state";
    $("#runState").textContent = "运行失败";
    $("#taskBadge").textContent = "失败";
  } finally {
    running = false;
    clearInterval(timer);
    $("#sendButton").disabled = false;
    $("#sendButton span").textContent = "继续研究";
    messages.scrollTop = messages.scrollHeight;
  }
}

examples.forEach((item) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "example-item";
  button.innerHTML = `${item.title}<small>${item.meta}</small>`;
  button.onclick = () => {
    $("#prompt").value = item.prompt;
    $("#prompt").focus();
  };
  $("#examples").appendChild(button);
});

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = $("#prompt").value.trim();
  if (!prompt) return;
  if (!runtimeConfig) {
    openConfig();
    return;
  }
  $("#prompt").value = "";
  submitResearch(prompt);
});
$("#prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
$("#configForm").addEventListener("submit", (event) => {
  event.preventDefault();
  saveConfig();
});
$("#provider").addEventListener("change", updateProviderDefaults);
$("#searchApi").addEventListener("change", updateSearchFields);
$("#openConfig").addEventListener("click", openConfig);
$("#closeConfig").addEventListener("click", () => $("#configDialog").close());
$("#clearKeys").addEventListener("click", () => {
  runtimeConfig = null;
  $("#modelKey").value = "";
  $("#tavilyKey").value = "";
  $("#openConfig span").textContent = "配置 Key";
  $("#openConfig").classList.remove("ready");
});
$("#newTask").onclick = () => window.location.reload();
$("#clearChat").onclick = () => window.location.reload();
$("#themeToggle").onclick = () => document.documentElement.classList.toggle("light");

updateSearchFields();
checkConnection();
setInterval(checkConnection, 30_000);
window.setTimeout(openConfig, 250);
