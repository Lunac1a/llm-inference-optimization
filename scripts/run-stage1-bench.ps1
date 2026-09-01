[CmdletBinding()]
param(
    [int]$Repetitions = 5,
    [int[]]$Depths = @(512, 4096, 16384, 32768),
    [int]$GeneratedTokens = 128,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$RunSet = 'bench'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bench = Join-Path $repo 'engine\llama.cpp\build-baseline\bin\llama-bench.exe'
$model = Join-Path $repo 'models\Qwen3-8B-Q4_K_M.gguf'
$lock = Get-Content -Raw (Join-Path $repo 'locks\model.json') | ConvertFrom-Json
$outDir = Join-Path $repo "artifacts\stage1\$RunSet"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not (Test-Path -LiteralPath $bench)) { throw "Missing llama-bench: $bench" }
if (-not (Test-Path -LiteralPath $model)) { throw "Missing model: $model" }

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash.ToLowerInvariant()
if ($actualHash -ne $lock.sha256) {
    throw "Model SHA-256 mismatch: expected $($lock.sha256), got $actualHash"
}

$engineHead = (& git -C (Join-Path $repo 'engine\llama.cpp') rev-parse HEAD).Trim()
$upstreamLock = Get-Content -Raw (Join-Path $repo 'locks\upstream.json') | ConvertFrom-Json
if ($engineHead -ne $upstreamLock.commit) {
    throw "Engine commit mismatch: expected $($upstreamLock.commit), got $engineHead"
}

$manifest = [ordered]@{
    started_at = (Get-Date).ToString('o')
    engine_commit = $engineHead
    model_sha256 = $actualHash
    repetitions = $Repetitions
    depths = $Depths
    generated_tokens = $GeneratedTokens
    run_set = $RunSet
    batch_size = 2048
    ubatch_size = 512
    threads = 24
    flash_attention = 'on'
    gpu_layers = -1
    kv_types = @('f16', 'q8_0')
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $outDir 'manifest.json')

function Invoke-OneRun {
    param([int]$Depth, [string]$KvType, [int]$Run)

    $stem = "depth-$Depth-$KvType-run-$Run"
    $stdout = Join-Path $outDir "$stem.json"
    $stderr = Join-Path $outDir "$stem.stderr.log"
    $telemetry = Join-Path $outDir "$stem.gpu.csv"
    $args = @(
        '-m', $model,
        '-p', '0',
        '-n', "$GeneratedTokens",
        '-d', "$Depth",
        '-ctk', $KvType,
        '-ctv', $KvType,
        '-ngl', '-1',
        '-fa', 'on',
        '-b', '2048',
        '-ub', '512',
        '-t', '24',
        '--poll', '50',
        '-r', '1',
        '-o', 'json'
    )

    'timestamp,temperature_c,pstate,power_w,sm_clock_mhz,memory_clock_mhz,gpu_util_pct,memory_util_pct,memory_used_mib' |
        Set-Content -Encoding ascii $telemetry

    $process = Start-Process -FilePath $bench -ArgumentList $args -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

    while (-not $process.HasExited) {
        $now = (Get-Date).ToString('o')
        $sample = & nvidia-smi --query-gpu=temperature.gpu,pstate,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        "$now,$sample" | Add-Content -Encoding ascii $telemetry
        Start-Sleep -Milliseconds 500
        $process.Refresh()
    }

    if ($process.ExitCode -ne 0) {
        throw "Benchmark failed for depth=$Depth kv=$KvType run=$Run; see $stderr"
    }
    $parsed = Get-Content -Raw $stdout | ConvertFrom-Json
    if ($parsed.Count -ne 1) { throw "Unexpected JSON record count in $stdout" }
    if ($parsed[0].build_commit -ne 'f8dbcd618') { throw "Unexpected build commit in $stdout" }
    if ($parsed[0].type_k -ne $KvType -or $parsed[0].type_v -ne $KvType) { throw "KV type mismatch in $stdout" }
    if ($parsed[0].flash_attn -ne 1) { throw "Flash Attention mismatch in $stdout" }
    if ($parsed[0].n_gpu_layers -ne -1) { throw "GPU offload mismatch in $stdout" }
}

foreach ($depth in $Depths) {
    for ($run = 1; $run -le $Repetitions; $run++) {
        $order = if ($run % 2 -eq 1) { @('f16', 'q8_0') } else { @('q8_0', 'f16') }
        foreach ($kv in $order) {
            Write-Host "Running depth=$depth kv=$kv repetition=$run"
            Invoke-OneRun -Depth $depth -KvType $kv -Run $run
        }
    }
}

$manifest.completed_at = (Get-Date).ToString('o')
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $outDir 'manifest.json')
Write-Host "Stage 1 microbenchmark raw data written to $outDir"
