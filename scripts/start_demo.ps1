$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\langgraph.exe')) {
    throw '未找到虚拟环境。请先在项目根目录运行 "uv sync"。'
}

if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Warning '已创建 .env。请填写模型和 Search API Key，然后重新运行此脚本。'
    exit 1
}

$envContent = Get-Content -LiteralPath '.env'
$hasModelKey = [bool]($envContent | Where-Object { $_ -match '^DEEPSEEK_API_KEY=.+' })

if (-not $hasModelKey) {
    Write-Warning '未找到 DEEPSEEK_API_KEY。服务可以启动，但无法生成真实报告。'
}

Write-Host '正在启动 CompeteX：http://127.0.0.1:2024'
Write-Host 'Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024'
& '.\.venv\Scripts\langgraph.exe' dev --no-browser --allow-blocking --no-reload --port 2024
