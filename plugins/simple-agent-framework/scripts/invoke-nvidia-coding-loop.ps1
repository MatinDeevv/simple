param(
  [Parameter(Mandatory=$true)][string]$Task,
  [Parameter(Mandatory=$true)][string]$Worktree,
  [Parameter(Mandatory=$true)][string[]]$Context,
  [Parameter(Mandatory=$true)][string[]]$Test,
  [Parameter(Mandatory=$true)][string[]]$AllowPath
)
$ErrorActionPreference = 'Stop'
if (!$env:NVIDIA_API_KEY) { [Console]::Error.WriteLine('NVIDIA_API_KEY is missing'); exit 2 }
$arguments = @((Join-Path $PSScriptRoot 'nvidia_coding_loop.py'), '--task', $Task, '--worktree', $Worktree)
foreach ($path in $Context) { $arguments += @('--context', $path) }
foreach ($command in $Test) { $arguments += @('--test', $command) }
foreach ($path in $AllowPath) { $arguments += @('--allow-path', $path) }
& python @arguments
exit $LASTEXITCODE
