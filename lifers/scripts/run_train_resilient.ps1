# Lifers 抗断电/崩溃 自动续接训练脚本 (PowerShell)
# 用法: powershell -ExecutionPolicy Bypass -File run_train_resilient.ps1
# 或注册为计划任务实现开机自启

$ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $ROOT

$CTL = "weights\.train_control"
$LOG = "weights\_resilient_loop.log"
$CRASH_LOG = "weights\_crash_history.log"

$env:PYTHONPATH = $ROOT
if (-not $env:LIFERS_ESCALATE_UNLIMITED) { $env:LIFERS_ESCALATE_UNLIMITED = "1" }
if (-not $env:LIFERS_RAMP_MAX_ITERS) { $env:LIFERS_RAMP_MAX_ITERS = "999999" }
if (-not $env:LIFERS_ESCALATE_RESUME) { $env:LIFERS_ESCALATE_RESUME = "1" }
if (-not $env:PYTHONUNBUFFERED) { $env:PYTHONUNBUFFERED = "1" }

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[${ts}] ===== Lifers 弹性训练启动 (PS) =====" | Out-File -Append $LOG -Encoding utf8
"  ROOT=$ROOT" | Out-File -Append $LOG -Encoding utf8

if (-not (Test-Path $CTL)) { "run" | Out-File -FilePath $CTL -Encoding ascii }

$CRASH_COUNT = 0
$BACKOFF = 2

while ($true) {
    if (Test-Path $CTL) {
        $MODE = (Get-Content $CTL -First 1).Trim().ToLower()
        if ($MODE -eq "stop") {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            "[${ts}] control=stop — 退出循环" | Out-File -Append $LOG -Encoding utf8
            exit 0
        }
        if ($MODE -eq "pause") {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            "[${ts}] control=pause — 等待中..." | Out-File -Append $LOG -Encoding utf8
            Start-Sleep -Seconds 10
            continue
        }
    }

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[${ts}] 启动训练 (崩溃计数: ${CRASH_COUNT})" | Out-File -Append $LOG -Encoding utf8

    try {
        $proc = Start-Process -FilePath "python" -ArgumentList "scripts/train_lifers_escalate.py" `
            -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$LOG.append" -RedirectStandardError "$LOG.append"
        $EXIT_CODE = $proc.ExitCode

        if (Test-Path "$LOG.append") {
            Get-Content "$LOG.append" | Out-File -Append $LOG -Encoding utf8
            Remove-Item "$LOG.append" -Force
        }

        if ($EXIT_CODE -eq 0) {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            "[${ts}] 训练正常结束 (exit 0)" | Out-File -Append $LOG -Encoding utf8
            $CRASH_COUNT = 0
            $BACKOFF = 2
            Start-Sleep -Seconds 3
            continue
        }
    } catch {
        $EXIT_CODE = -1
    }

    $CRASH_COUNT++
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[${ts}] 崩溃! exit=${EXIT_CODE} 连续: ${CRASH_COUNT}" | Out-File -Append $CRASH_LOG -Encoding utf8

    $BACKOFF = switch ($CRASH_COUNT) {
        1 { 2 }
        2 { 4 }
        3 { 8 }
        4 { 16 }
        default { 30 }
    }

    "[${ts}] ${BACKOFF}s 后重试..." | Out-File -Append $LOG -Encoding utf8
    Start-Sleep -Seconds $BACKOFF
}
