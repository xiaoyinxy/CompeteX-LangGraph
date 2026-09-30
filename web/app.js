const API_BASE = "http://127.0.0.1:2030";
const ASSISTANT_ID = "CompeteX Research Agent";

const examples = [
  { title: "AI 会议助手", meta: "市场进入与 MVP 机会", prompt: "请为计划进入中国市场的 AI 会议助手团队，对比 Otter.ai、Fireflies.ai 和 Notion AI Meeting Notes。重点覆盖目标用户、核心工作流、定价、集成生态、差异化、近期产品动态和用户采用阻力。区分事实、推断与建议，最后给出面向中国科技公司的 MVP 机会清单。" },
  { title: "AI 编程工具", meta: "产品定位与差异化", prompt: "对比 Cursor、GitHub Copilot 与 Windsurf 的目标用户、核心功能、定价和差异化，分析中国开发者工具团队可切入的机会。重要事实请保留来源。" },
  { title: "智能客服平台", meta: "企业采购决策", prompt: "以一家 200 人电商企业的采购视角，对比三类主流 AI 智能客服方案，重点分析部署成本、知识库效果、人工接管、安全合规与实施风险。" }
];

const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
let threadId = null;
let running = false;
let timer = null;
let startTime = 0;
let completedNodes = new Set();

function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c])); }
function renderMarkdown(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  safe = safe.replace(/^### (.+)$/gm, '<h3>$1</h3>').replace(/^## (.+)$/gm, '<h2>$1</h2>').replace(/^# (.+)$/gm, '<h1>$1</h1>');
  safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/^[-*] (.+)$/gm, '<li>$1</li>').replace(/(?:<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
  return safe.split(/\n{2,}/).map(block => /^<(h\d|ul)/.test(block) ? block : `<p>${block.replace(/\n/g,"<br>")}</p>`).join("");
}
function addMessage(role, content, loading=false) {
  const item = document.createElement("article"); item.className = `message ${role}`;
  item.innerHTML = `<div class="avatar">${role === "user" ? "我" : "CX"}</div><div class="bubble">${loading ? '<div class="thinking"><i class="spinner"></i><span>正在理解需求并规划研究路径…</span></div>' : renderMarkdown(content)}</div>`;
  messages.appendChild(item); messages.scrollTop = messages.scrollHeight; return item.querySelector(".bubble");
}
function resetRun() {
  completedNodes.clear(); clearInterval(timer); $("#elapsed").textContent="00:00"; $("#sourceCount").textContent="0"; $("#runState").className="run-state"; $("#runState").textContent="待开始"; $("#taskBadge").textContent="待开始";
  document.querySelectorAll(".timeline li").forEach(li => {li.className="";li.querySelector("time").textContent="待处理"}); updateProgress();
}
function updateProgress() {
  const count = completedNodes.size, percent = Math.round(count / 4 * 100); $("#progressBar").style.width=`${percent}%`; $("#progressText").textContent=`${percent}%`; $("#nodeCount").textContent=`${count}/4`;
}
function markNode(name, done=false) {
  const aliases = {
    clarify_with_user:"clarify_with_user",
    write_research_brief:"write_research_brief",
    research_supervisor:"research_supervisor",
    supervisor:"research_supervisor",
    researcher:"research_supervisor",
    compress_research:"research_supervisor",
    final_report_generation:"final_report_generation"
  };
  const match = Object.keys(aliases).find(k => name.includes(k)); if (!match) return;
  const key = aliases[match];
  const li = document.querySelector(`[data-node="${key}"]`); if (done) {li.className="done";li.querySelector("time").textContent="已完成";completedNodes.add(key)} else {li.className="active";li.querySelector("time").textContent="执行中"} updateProgress();
}
async function verifyRunOutcome(answer) {
  const stateRes = await fetch(`${API_BASE}/threads/${threadId}/state`);
  if (!stateRes.ok) throw new Error(`无法读取运行状态（${stateRes.status}）`);
  const state = await stateRes.json();
  const report = state?.values?.final_report;
  if (report) {
    answer.innerHTML = renderMarkdown(report);
    for (const key of ["clarify_with_user","write_research_brief","research_supervisor","final_report_generation"]) markNode(key, true);
    $("#sourceCount").textContent = String((report.match(/https?:\/\//g)||[]).length);
    return "complete";
  }

  const runsRes = await fetch(`${API_BASE}/threads/${threadId}/runs`);
  const runs = runsRes.ok ? await runsRes.json() : [];
  const latestRun = Array.isArray(runs) ? runs[0] : null;
  if (latestRun?.status === "error") {
    let detail = "LangGraph 后端运行失败";
    try {
      const joined = await fetch(`${API_BASE}/threads/${threadId}/runs/${latestRun.run_id}/join`).then(r => r.json());
      detail = joined?.__error__?.message || joined?.__error__?.error || detail;
    } catch {}
    throw new Error(detail);
  }

  const stateMessages = state?.values?.messages || [];
  const lastAssistant = [...stateMessages].reverse().find(message => {
    const type = message?.type || message?.role;
    return type === "ai" || type === "assistant";
  });
  const clarification = extractText(lastAssistant);
  if (clarification) answer.innerHTML = renderMarkdown(clarification);
  return "awaiting_input";
}
function extractText(value) {
  if (!value) return ""; if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(extractText).join("");
  if (value.content) return extractText(value.content); if (value.text) return value.text;
  return "";
}
function friendlyErrorMessage(error) {
  const message = String(error?.message || error || "未知错误");
  if (/APIConnectionError|internal error occurred/i.test(message)) {
    return "DeepSeek API 连接失败。请检查 API Key、网络连接和 DeepSeek 服务状态后重试。";
  }
  if (/Failed to fetch|NetworkError|无法读取运行状态|创建会话失败|研究请求失败/i.test(message)) {
    return "无法连接本地 LangGraph 服务。请确认 PowerShell 中的服务仍在运行。";
  }
  return message;
}
function consumeEvent(eventName, payload, answer) {
  if (eventName === "error") {
    const detail = payload?.message || payload?.error || "LangGraph 运行失败";
    throw new Error(detail);
  }
  if (Array.isArray(payload) && payload.length === 2 && typeof payload[0] === "string") {
    eventName = payload[0]; payload = payload[1];
  }
  if (eventName === "updates" && payload && typeof payload === "object") {
    for (const [node, output] of Object.entries(payload)) {
      markNode(node, true);
      const text = output?.final_report || extractText(output?.messages);
      if (text) answer.innerHTML = renderMarkdown(text);
    }
  }
  const nodeName = payload?.name || payload?.metadata?.langgraph_node || payload?.data?.name || "";
  if (eventName.includes("start")) markNode(nodeName, false); if (eventName.includes("end")) markNode(nodeName, true);
  const data = payload?.data ?? payload;
  const report = data?.output?.final_report || data?.final_report || data?.chunk?.final_report;
  const msg = data?.chunk?.messages || data?.output?.messages;
  const text = report || extractText(msg);
  if (text && text.length >= answer.textContent.length) answer.innerHTML = renderMarkdown(text);
  const combined = report || answer.textContent; if (combined) $("#sourceCount").textContent = String((combined.match(/https?:\/\//g)||[]).length);
}
async function checkConnection() {
  const el=$("#connection"); try {const res=await fetch(`${API_BASE}/ok`); if(!res.ok) throw new Error(); el.className="connection online";el.querySelector("span").textContent="本地服务已连接"} catch {el.className="connection offline";el.querySelector("span").textContent="本地服务未启动"}
}
async function submitResearch(prompt) {
  if(running) return; running=true; resetRun(); $("#sendButton").disabled=true; $("#sendButton span").textContent="研究进行中"; $("#runState").className="run-state running";$("#runState").textContent="运行中";$("#taskBadge").textContent="进行中";
  document.querySelector(".welcome-card")?.remove(); addMessage("user",prompt); const answer=addMessage("assistant","",true); startTime=Date.now(); timer=setInterval(()=>{const sec=Math.floor((Date.now()-startTime)/1000);$("#elapsed").textContent=`${String(Math.floor(sec/60)).padStart(2,"0")}:${String(sec%60).padStart(2,"0")}`},1000);
  try {
    if(!threadId){const threadRes=await fetch(`${API_BASE}/threads`,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});if(!threadRes.ok)throw new Error(`创建会话失败（${threadRes.status}）`);threadId=(await threadRes.json()).thread_id}
    const response=await fetch(`${API_BASE}/threads/${threadId}/runs/stream`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({assistant_id:ASSISTANT_ID,input:{messages:[{role:"user",content:prompt}]},stream_mode:["updates","values"]})});
    if(!response.ok) throw new Error(`研究请求失败（${response.status}）：${await response.text()}`);
    answer.innerHTML='<div class="thinking"><i class="spinner"></i><span>研究节点正在运行，结果将实时呈现…</span></div>';
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer="",eventName="message";
    while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split(/\r?\n/);buffer=lines.pop();for(const line of lines){if(line.startsWith("event:"))eventName=line.slice(6).trim();else if(line.startsWith("data:")){let parsed;try{parsed=JSON.parse(line.slice(5))}catch{continue}consumeEvent(eventName,parsed,answer)}}}
    const outcome = await verifyRunOutcome(answer);
    if (outcome === "complete") {
      updateProgress();$("#runState").className="run-state done";$("#runState").textContent="已完成";$("#taskBadge").textContent="已完成";$("#interviewTip").textContent="演示时强调：最终建议并非模型直觉，而是由多轮检索证据和覆盖度反思逐步收敛而来。";
    } else {
      $("#runState").className="run-state running";$("#runState").textContent="等待补充";$("#taskBadge").textContent="待补充";$("#interviewTip").textContent="Agent 正在澄清研究范围。回答中间的问题后，将在同一会话中继续研究。";
    }
  } catch(error) { answer.innerHTML=`<p><strong>运行失败</strong></p><p>${escapeHtml(friendlyErrorMessage(error))}</p>`;$("#runState").className="run-state";$("#runState").textContent="运行失败";$("#taskBadge").textContent="失败" }
  finally {running=false;clearInterval(timer);$("#sendButton").disabled=false;$("#sendButton span").textContent="继续研究";messages.scrollTop=messages.scrollHeight}
}

examples.forEach(item=>{const b=document.createElement("button");b.type="button";b.className="example-item";b.innerHTML=`${item.title}<small>${item.meta}</small>`;b.onclick=()=>{$("#prompt").value=item.prompt;$("#prompt").focus()};$("#examples").appendChild(b)});
$("#composer").addEventListener("submit",e=>{e.preventDefault();const p=$("#prompt").value.trim();if(p){$("#prompt").value="";submitResearch(p)}});
$("#prompt").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();$("#composer").requestSubmit()}});
$("#newTask").onclick=()=>location.reload();$("#clearChat").onclick=()=>location.reload();
$("#themeToggle").onclick=()=>document.documentElement.classList.toggle("light");checkConnection();setInterval(checkConnection,15000);
