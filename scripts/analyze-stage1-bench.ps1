[CmdletBinding()]
param(
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$RunSet = 'bench'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$inDir = Join-Path $repo "artifacts\stage1\$RunSet"
$outFile = Join-Path $repo "artifacts\stage1\$RunSet-summary.csv"

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    $n = $sorted.Count
    if ($n % 2 -eq 1) { return [double]$sorted[[int][math]::Floor($n / 2)] }
    return ([double]$sorted[$n / 2 - 1] + [double]$sorted[$n / 2]) / 2.0
}

$rows = foreach ($group in (Get-ChildItem $inDir -Filter 'depth-*-run-*.json' | ForEach-Object {
    $record = (Get-Content -Raw $_.FullName | ConvertFrom-Json)[0]
    [pscustomobject]@{
        depth = [int]$record.n_depth
        kv_type = [string]$record.type_k
        tokens_per_second = [double]$record.avg_ts
        elapsed_ns = [long]$record.avg_ns
        source_file = $_.Name
    }
} | Group-Object depth, kv_type)) {
    $values = [double[]]@($group.Group.tokens_per_second)
    $mean = ($values | Measure-Object -Average).Average
    $sumSq = 0.0
    foreach ($value in $values) { $sumSq += [math]::Pow($value - $mean, 2) }
    $stddev = if ($values.Count -gt 1) { [math]::Sqrt($sumSq / ($values.Count - 1)) } else { 0.0 }
    [pscustomobject]@{
        depth = $group.Group[0].depth
        kv_type = $group.Group[0].kv_type
        repetitions = $values.Count
        median_tokens_per_second = [math]::Round((Get-Median $values), 6)
        mean_tokens_per_second = [math]::Round($mean, 6)
        stddev_tokens_per_second = [math]::Round($stddev, 6)
        cv_percent = [math]::Round(100.0 * $stddev / $mean, 4)
        min_tokens_per_second = [math]::Round(($values | Measure-Object -Minimum).Minimum, 6)
        max_tokens_per_second = [math]::Round(($values | Measure-Object -Maximum).Maximum, 6)
    }
}

$rows | Sort-Object depth, kv_type | Export-Csv -NoTypeInformation -Encoding utf8 $outFile
$rows | Sort-Object depth, kv_type | Format-Table -AutoSize
Write-Host "Summary written to $outFile"
