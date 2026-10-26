# Run manually when ready to download dependencies. This script never starts training.
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$PythonExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$PythonArguments)
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE. Environment setup is incomplete."
    }
}

if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    throw 'Pass the absolute path to a complete 64-bit CPython 3.11 or 3.12 python.exe.'
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable does not exist: $PythonExe"
}
$resolvedPython = (Resolve-Path -LiteralPath $PythonExe).ProviderPath
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvRoot = Join-Path $projectRoot '.venv'
$requirementsPath = Join-Path $projectRoot 'requirements-train.txt'
if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
    throw "Missing requirements file: $requirementsPath"
}

# Importing venv/ensurepip verifies these standard-library components exist;
# neither import installs packages, initializes CUDA, or creates an environment.
$probeCode = "import sys, struct, json, venv, ensurepip; print(json.dumps({'implementation': sys.implementation.name, 'major': sys.version_info.major, 'minor': sys.version_info.minor, 'bits': struct.calcsize('P') * 8}))"
$probeText = & $resolvedPython -I -c $probeCode
if ($LASTEXITCODE -ne 0) {
    throw 'The selected Python is incomplete. Use a full CPython installation with venv and ensurepip, not an embedded runtime.'
}
$probe = ($probeText -join "`n") | ConvertFrom-Json
if ($probe.implementation -ne 'cpython' -or $probe.major -ne 3 -or $probe.minor -notin @(11, 12) -or $probe.bits -ne 64) {
    throw 'This setup requires a complete 64-bit CPython 3.11 or 3.12 installation.'
}

if (Test-Path -LiteralPath $venvRoot) {
    $venvItem = Get-Item -LiteralPath $venvRoot -Force
    if (-not $venvItem.PSIsContainer -or ($venvItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw 'The project .venv must be a regular directory, not a file, junction, or symbolic link.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $venvRoot 'pyvenv.cfg') -PathType Leaf)) {
        throw 'Existing .venv is not a recognized virtual environment. It was left untouched.'
    }
}

Write-Host "Project: $projectRoot"
Write-Host "Interpreter: $resolvedPython"
Write-Host 'Planned: local .venv, torch 2.8.0 from the official cu126 index, requirements-train.txt.'
Write-Host 'This downloads dependencies. It does not change GPU drivers or start model/data downloads or training.'
if (-not $PSCmdlet.ShouldProcess($venvRoot, 'Create or reuse a project virtual environment and install training dependencies')) {
    return
}

if (-not (Test-Path -LiteralPath $venvRoot)) {
    Invoke-CheckedPython -Executable $resolvedPython -PythonArguments @('-I', '-m', 'venv', $venvRoot)
}
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw 'The project virtual environment has no Scripts\python.exe. It was not deleted or replaced.'
}
$venvProbeText = & $venvPython -I -c $probeCode
if ($LASTEXITCODE -ne 0) {
    throw 'The existing virtual environment is incomplete. It was left untouched.'
}
$venvProbe = ($venvProbeText -join "`n") | ConvertFrom-Json
if ($venvProbe.implementation -ne 'cpython' -or $venvProbe.major -ne 3 -or $venvProbe.minor -notin @(11, 12) -or $venvProbe.bits -ne 64) {
    throw 'Existing .venv does not use supported 64-bit CPython 3.11/3.12. It was left untouched.'
}

Invoke-CheckedPython -Executable $venvPython -PythonArguments @('-m', 'pip', 'install', 'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu126')
Invoke-CheckedPython -Executable $venvPython -PythonArguments @('-m', 'pip', 'install', '-r', $requirementsPath)
Invoke-CheckedPython -Executable $venvPython -PythonArguments @('-m', 'pip', 'check')

Write-Host 'Dependencies installed and pip check passed. GPU training compatibility still requires a separate smoke run.'
Write-Host "Run from $projectRoot using .\.venv\Scripts\python.exe -m rlhf_lab doctor"
