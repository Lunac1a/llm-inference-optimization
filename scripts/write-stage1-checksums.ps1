[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = Join-Path $repo 'artifacts\stage1'
$output = Join-Path $root 'SHA256SUMS.txt'
$lines = Get-ChildItem -LiteralPath $root -Recurse -File |
    Where-Object { $_.FullName -ne $output } |
    ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($repo, $_.FullName).Replace('\', '/')
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        "$hash  $relative"
    } |
    Sort-Object
[System.IO.File]::WriteAllLines($output, [string[]]$lines, [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote $($lines.Count) checksums to $output"
