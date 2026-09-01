[CmdletBinding()]
param(
    [int]$Repetitions = 3,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$RunSet = 'e2e'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cli = Join-Path $repo 'engine\llama.cpp\build-baseline\bin\llama-cli.exe'
$model = Join-Path $repo 'models\Qwen3-8B-Q4_K_M.gguf'
$outDir = Join-Path $repo "artifacts\stage1\$RunSet"
$promptFile = Join-Path $outDir 'fixed-prompt.txt'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$builder = [System.Text.StringBuilder]::new()
[void]$builder.AppendLine('Read the following deterministic records, then continue with a concise technical summary.')
for ($i = 1; $i -le 400; $i++) {
    [void]$builder.AppendLine(('Record {0:D4}: alpha beta gamma delta epsilon zeta eta theta; preserve order and identify recurring structure.' -f $i))
}
[System.IO.File]::WriteAllText($promptFile, $builder.ToString(), [System.Text.UTF8Encoding]::new($false))
$promptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $promptFile).Hash.ToLowerInvariant()

$manifest = [ordered]@{
    started_at = (Get-Date).ToString('o')
    prompt_sha256 = $promptHash
    run_set = $RunSet
    prompt_recipe = 'header plus 400 numbered deterministic ASCII records'
    repetitions = $Repetitions
    context_size = 16384
    generated_tokens = 128
    seed = 42
    temperature = 0.0
    top_k = 1
    top_p = 1.0
    min_p = 0.0
    repeat_penalty = 1.0
    ignore_eos = $true
    batch_size = 2048
    ubatch_size = 512
    threads = 24
    flash_attention = 'on'
    gpu_layers = 'all'
    kv_types = @('f16', 'q8_0')
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $outDir 'manifest.json')

function Invoke-OneRun([string]$KvType, [int]$Run) {
    $stem = "$KvType-run-$Run"
    $stdout = Join-Path $outDir "$stem.stdout.txt"
    $stderr = Join-Path $outDir "$stem.stderr.log"
    $telemetry = Join-Path $outDir "$stem.gpu.csv"
    $args = @(
        '-m', $model,
        '-f', $promptFile,
        '-n', '128',
        '-c', '16384',
        '-b', '2048',
        '-ub', '512',
        '-t', '24',
        '--poll', '50',
        '-ngl', 'all',
        '-fa', 'on',
        '-ctk', $KvType,
        '-ctv', $KvType,
        '-s', '42',
        '--temp', '0',
        '--top-k', '1',
        '--top-p', '1',
        '--min-p', '0',
        '--repeat-penalty', '1',
        '--ignore-eos',
        '--no-display-prompt',
        '--simple-io',
        '--single-turn',
        '-lv', '4'
    )

    'timestamp,temperature_c,pstate,power_w,sm_clock_mhz,memory_clock_mhz,gpu_util_pct,memory_util_pct,memory_used_mib' |
        Set-Content -Encoding ascii $telemetry
    $process = Start-Process -FilePath $cli -ArgumentList $args -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    while (-not $process.HasExited) {
        $now = (Get-Date).ToString('o')
        $sample = & nvidia-smi --query-gpu=temperature.gpu,pstate,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        "$now,$sample" | Add-Content -Encoding ascii $telemetry
        Start-Sleep -Milliseconds 500
        $process.Refresh()
    }
    if ($process.ExitCode -ne 0) { throw "E2E run failed for kv=$KvType run=$Run; see $stderr" }
    $log = [string]::Join("`n", @(
        [string](Get-Content -Raw $stdout),
        [string](Get-Content -Raw $stderr)
    ))
    if ($log -notmatch 'flash_attn\s+= enabled') { throw "Flash Attention evidence missing in $stderr" }
    if ($log -notmatch 'offloaded 37/37 layers to GPU') { throw "Full GPU offload evidence missing in $stderr" }
    if ($log -notmatch "K \($KvType\)" -or $log -notmatch "V \($KvType\)") { throw "Runtime KV type evidence missing for $KvType" }
    if ($log -notmatch 'top_k = 1, top_p = 1\.000, min_p = 0\.000') { throw "Sampler evidence missing in $stderr" }
    if ($log -notmatch 'eval time') { throw "Decode timing missing in $stderr" }
}

for ($run = 1; $run -le $Repetitions; $run++) {
    $order = if ($run % 2 -eq 1) { @('f16', 'q8_0') } else { @('q8_0', 'f16') }
    foreach ($kv in $order) {
        Write-Host "Running end-to-end kv=$kv repetition=$run"
        Invoke-OneRun -KvType $kv -Run $run
    }
}

$manifest.completed_at = (Get-Date).ToString('o')
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $outDir 'manifest.json')
Write-Host "Stage 1 end-to-end raw data written to $outDir"
