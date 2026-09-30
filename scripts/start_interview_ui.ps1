$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\langgraph.exe')) {
    throw '未找到虚拟环境。请先在项目根目录运行 "uv sync"。'
}

if (-not (Test-Path -LiteralPath '.env')) {
    throw '未找到 .env。请将 .env.example 复制为 .env，并填写 DEEPSEEK_API_KEY。'
}

function Test-HttpEndpoint {
    param([string]$Uri, [string]$ExpectedContent = '')
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ne 200) { return $false }
        if ($ExpectedContent -and $response.Content -notmatch [regex]::Escape($ExpectedContent)) { return $false }
        return $true
    } catch {
        return $false
    }
}

$apiJob = $null
$apiAlreadyRunning = $false
$uiAlreadyRunning = $false

$apiListener = Get-NetTCPConnection -LocalPort 2030 -State Listen -ErrorAction SilentlyContinue
if ($apiListener) {
    if (Test-HttpEndpoint -Uri 'http://127.0.0.1:2030/ok') {
        $apiAlreadyRunning = $true
        Write-Host '复用正在运行的 CompeteX API：http://127.0.0.1:2030'
    } else {
        throw '端口 2030 已被其他服务或状态异常的服务占用。'
    }
}

$uiListener = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($uiListener) {
    if (Test-HttpEndpoint -Uri 'http://127.0.0.1:3000/' -ExpectedContent 'CompeteX') {
        $uiAlreadyRunning = $true
        Write-Host '复用正在运行的中文 UI：http://127.0.0.1:3000'
    } else {
        throw '端口 3000 已被其他服务占用。'
    }
}

if (-not $apiAlreadyRunning) {
    $apiJob = Start-Job -ScriptBlock {
        param($root)
        Set-Location -LiteralPath $root
        $env:PYTHONUTF8 = '1'
        # 中文 UI 使用固定端口长期运行；关闭热重载可避免 Windows 下残留子进程继续占用 2030，
        # 也避免开发期文件监听反复重载正在执行的长时研究任务。
        & '.\.venv\Scripts\langgraph.exe' dev --no-browser --allow-blocking --no-reload --port 2030
    } -ArgumentList $projectRoot
}

try {
    if (-not $apiAlreadyRunning) {
        Write-Host '正在启动 CompeteX API：http://127.0.0.1:2030'
        $ready = $false
        for ($attempt = 0; $attempt -lt 40; $attempt++) {
            Start-Sleep -Milliseconds 500
            if (Test-HttpEndpoint -Uri 'http://127.0.0.1:2030/ok') { $ready = $true; break }
            if ($apiJob.State -eq 'Failed') { break }
        }
        if (-not $ready) {
            Receive-Job -Job $apiJob
            throw 'CompeteX API 启动失败。'
        }
    }

    Write-Host '中文面试界面：http://127.0.0.1:3000'
    Start-Process 'http://127.0.0.1:3000'
    if ($apiAlreadyRunning -and $uiAlreadyRunning) {
        Write-Host 'CompeteX 已经在运行，现有服务保持不变。'
        exit 0
    }
    Write-Host '按 Ctrl+C 可停止本脚本启动的服务。'
    if (-not $uiAlreadyRunning) {
        & python -m http.server 3000 --bind 127.0.0.1 --directory web
    } else {
        while ($apiJob.State -eq 'Running') { Start-Sleep -Seconds 1 }
        Receive-Job -Job $apiJob
    }
} finally {
    if ($apiJob) {
        Stop-Job -Job $apiJob -ErrorAction SilentlyContinue
        Remove-Job -Job $apiJob -Force -ErrorAction SilentlyContinue
    }
}
