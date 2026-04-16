param(
    [ValidateSet(
        "preprocess",
        "knowledge_base",
        "bert_classifier",
        "ensemble",
        "metrics",
        "ablation_100",
        "reviewer_audit"
    )]
    [string]$StartAt = "preprocess"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Expected virtualenv interpreter at $python"
}

$logDir = Join-Path $root "results\logs\liar_review_safe"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$completeMarker = Join-Path $logDir "COMPLETE.txt"
if (Test-Path $completeMarker) {
    Remove-Item $completeMarker -Force
}

# Reviewer-safe run profile
$env:DATASET = "liar"
$env:BENCHMARK_PROFILE = "verification"
$env:CHEAP_MODE = "false"
$env:STRICT_REVIEW_MODE = "true"
$env:SAVE_ALL_EVAL_RESULTS = "false"
$env:PYTHONUNBUFFERED = "1"

# LIAR claims are short, so 256 tokens is a pragmatic speed/quality tradeoff.
# Use a conservative batch size for laptop-GPU stability unless the caller
# explicitly overrides it in the environment before launching the script.
if (-not $env:BERT_MAX_LENGTH) { $env:BERT_MAX_LENGTH = "256" }
if (-not $env:BERT_BATCH_SIZE) { $env:BERT_BATCH_SIZE = "8" }
if (-not $env:BERT_EPOCHS) { $env:BERT_EPOCHS = "3" }
if (-not $env:ENSEMBLE_SAMPLE_SIZE) { $env:ENSEMBLE_SAMPLE_SIZE = "200" }
if (-not $env:OLLAMA_TIMEOUT_SEC) { $env:OLLAMA_TIMEOUT_SEC = "120" }
if (-not $env:LLM_MAX_RETRIES) { $env:LLM_MAX_RETRIES = "2" }

function Write-StepHeader {
    param(
        [string]$Name,
        [string]$LogPath
    )

    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "RUNNING: $Name"
    Write-Host "LOG: $LogPath"
    Write-Host ("=" * 70)
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$ArgumentList
    )

    $logPath = Join-Path $logDir "$Name.log"
    $stdoutPath = Join-Path $logDir "$Name.stdout.tmp"
    $stderrPath = Join-Path $logDir "$Name.stderr.tmp"

    foreach ($path in @($logPath, $stdoutPath, $stderrPath)) {
        if (Test-Path $path) {
            Remove-Item $path -Force
        }
    }

    Write-StepHeader -Name $Name -LogPath $logPath

    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList @("-u") + $ArgumentList `
        -WorkingDirectory $root `
        -PassThru `
        -Wait `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $combined = @()
    if (Test-Path $stdoutPath) {
        $combined += Get-Content $stdoutPath
    }
    if (Test-Path $stderrPath) {
        $stderr = Get-Content $stderrPath
        if ($stderr) {
            $combined += ""
            $combined += "[stderr]"
            $combined += $stderr
        }
    }

    $combined | Set-Content -Path $logPath -Encoding UTF8
    if ($combined) {
        Get-Content $logPath -Tail 40
    }

    Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

    if ($proc.ExitCode -ne 0) {
        throw "Step failed: $Name (exit code $($proc.ExitCode))"
    }
}

function Run-PythonCodeStep {
    param(
        [string]$Name,
        [string]$Code
    )

    $logPath = Join-Path $logDir "$Name.log"
    $tempCodePath = Join-Path $logDir "$Name.tmp.py"
    $Code | Set-Content -Path $tempCodePath -Encoding UTF8

    try {
        Invoke-PythonStep -Name $Name -ArgumentList @($tempCodePath)
    }
    finally {
        Remove-Item $tempCodePath -Force -ErrorAction SilentlyContinue
    }
}

$steps = @(
    @{
        Name = "preprocess"
        Type = "file"
        Value = @("-m", "src.preprocessing.preprocessor")
    },
    @{
        Name = "knowledge_base"
        Type = "file"
        Value = @("-m", "src.rag.knowledge_base_builder")
    },
    @{
        Name = "bert_classifier"
        Type = "file"
        Value = @("-m", "src.models.bert_classifier")
    },
    @{
        Name = "ensemble"
        Type = "file"
        Value = @("-m", "src.models.ensemble")
    },
    @{
        Name = "metrics"
        Type = "file"
        Value = @("-m", "src.evaluation.metrics")
    },
    @{
        Name = "ablation_100"
        Type = "code"
        Value = @'
from src.evaluation.ablation import run_ablation
run_ablation(sample_size=100)
'@
    },
    @{
        Name = "reviewer_audit"
        Type = "file"
        Value = @("-m", "src.evaluation.reviewer_audit")
    }
)

$started = $false
foreach ($step in $steps) {
    if ($step.Name -eq $StartAt) {
        $started = $true
    }

    if (-not $started) {
        continue
    }

    if ($step.Type -eq "file") {
        Invoke-PythonStep -Name $step.Name -ArgumentList $step.Value
    } else {
        Run-PythonCodeStep -Name $step.Name -Code $step.Value
    }
}

@"
Completed at: $(Get-Date -Format o)
StartAt: $StartAt
"@ | Set-Content -Path $completeMarker -Encoding UTF8

Write-Host ""
Write-Host "Reviewer-safe LIAR workflow finished."
Write-Host "Logs: $logDir"
Write-Host "Marker: $completeMarker"
