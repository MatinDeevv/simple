param(
  [Parameter(Mandatory=$true)][string]$Task,
  [Parameter(Mandatory=$true)][string]$Worktree,
  [Parameter(Mandatory=$true)][string[]]$Context,
  [Parameter(Mandatory=$true)][string[]]$Test
)
$ErrorActionPreference = 'Stop'
$envPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\.env'))
if (!$env:NVIDIA_API_KEY -and (Test-Path $envPath -PathType Leaf)) {
  $line = Get-Content $envPath | Where-Object { $_ -match '^NVIDIA_API_KEY=(.*)$' } | Select-Object -First 1
  if ($line) { $env:NVIDIA_API_KEY = ($line -split '=',2)[1].Trim().Trim('"').Trim("'") }
}
if (!$env:NVIDIA_API_KEY) { [Console]::Error.WriteLine('NVIDIA_API_KEY is missing'); exit 2 }
$arguments = @((Join-Path $PSScriptRoot 'nvidia_coding_loop.py'), '--task', $Task, '--worktree', $Worktree)
foreach ($path in $Context) { $arguments += @('--context', $path) }
foreach ($command in $Test) { $arguments += @('--test', $command) }
& python @arguments
exit $LASTEXITCODE
