#Requires -Version 5.1

<#
.SYNOPSIS
Runs DAMA training natively on Windows with a CUDA-enabled Conda environment.

.DESCRIPTION
This launcher bypasses WSL and its GPU bridge. It performs read-only preflight
checks before starting training and does not install packages, compile Cython
extensions, edit configuration files, or modify Git state.

Training itself still writes the logs, replay data, and checkpoints configured
by DAMA.

.PARAMETER CondaEnvironment
Name of the Windows Conda environment. The default is "dama".

.PARAMETER Config
Training configuration path relative to the DAMA repository. The default
matches local_train.sh.

.PARAMETER Resume
Resume from a specific checkpoint. This cannot be combined with FreshStart.

.PARAMETER FreshStart
Start without a checkpoint. Without this option or Resume, the launcher uses
the latest checkpoint, matching the current local_train.sh behavior.

.PARAMETER TrainDuration
Optional duration such as 2d, 4h, 30m, or 1d12h.

.PARAMETER ValidateOnly
Run all environment, dependency, project-import, and CUDA checks without
starting training.

.EXAMPLE
.\local_train.ps1 -ValidateOnly

.EXAMPLE
.\local_train.ps1

.EXAMPLE
.\local_train.ps1 -FreshStart -TrainDuration 4h

.EXAMPLE
.\local_train.ps1 -Resume .\models\checkpoints\model_step_10000.pt
#>

[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string] $CondaEnvironment = 'dama',

    [ValidateNotNullOrEmpty()]
    [string] $Config = 'config\training_config_policy_distillation.yaml',

    [string] $Resume = '',

    [switch] $FreshStart,

    [ValidatePattern('^(?=.+)(?:\d+d)?(?:\d+h)?(?:\d+m)?$')]
    [string] $TrainDuration = '',

    [switch] $ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw 'This launcher must run in Windows PowerShell or PowerShell on Windows.'
}

if ($FreshStart -and -not [string]::IsNullOrWhiteSpace($Resume)) {
    throw 'Use either -FreshStart or -Resume, not both.'
}

$ProjectDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$ProjectPrefix = $ProjectDirectory.TrimEnd('\') + '\'
$ConfigCandidate = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $ProjectDirectory $Config
}
$ConfigPath = [System.IO.Path]::GetFullPath($ConfigCandidate)

if (-not $ConfigPath.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The training config must be inside the DAMA repository: $ProjectDirectory"
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Training config not found: $ConfigPath"
}

$ResumePath = ''
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $ResumeCandidate = if ([System.IO.Path]::IsPathRooted($Resume)) {
        $Resume
    } else {
        Join-Path $ProjectDirectory $Resume
    }
    $ResumePath = [System.IO.Path]::GetFullPath($ResumeCandidate)
    if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) {
        throw "Checkpoint not found: $ResumePath"
    }
}

$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $CondaCommand) {
    throw 'Conda was not found on PATH. Open an Anaconda or Miniconda PowerShell prompt and try again.'
}
$CondaExecutable = $CondaCommand.Source

$LogDirectory = Join-Path $ProjectDirectory 'logs\local\console'
[void] [System.IO.Directory]::CreateDirectory($LogDirectory)
$LogTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDirectory "console_windows_$LogTimestamp.txt"

$Mutex = New-Object System.Threading.Mutex($false, 'Local\DAMA_Local_Train_Windows')
$MutexAcquired = $false
$TranscriptStarted = $false
$LocationPushed = $false

$EnvironmentNames = @(
    'PYTHONPATH',
    'PROCESS_TITLE',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_MAX_THREADS'
)
$SavedEnvironment = @{}
foreach ($Name in $EnvironmentNames) {
    $SavedEnvironment[$Name] = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
}

function Invoke-CondaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $SavedPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can convert native stderr into PowerShell errors.
        # Let the native process exit code determine success instead.
        $ErrorActionPreference = 'Continue'
        & $script:CondaExecutable @Arguments
        $script:NativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $SavedPreference
    }
}

try {
    try {
        $MutexAcquired = $Mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $MutexAcquired = $true
    }
    if (-not $MutexAcquired) {
        throw 'Another native Windows DAMA training launcher is already running.'
    }

    Start-Transcript -Path $LogFile -Append | Out-Null
    $TranscriptStarted = $true

    Write-Host '=== Filipino Dama - ML Training (Native Windows) ==='
    Write-Host "Project:   $ProjectDirectory"
    Write-Host "Config:    $ConfigPath"
    Write-Host "Conda env: $CondaEnvironment"
    Write-Host "Log:       $LogFile"
    Write-Host ''
    Write-Host 'Source safeguard: no package installation, source build, config edit, or Git mutation is performed.'
    Write-Host 'Expected training outputs may still be written according to the selected config.'
    Write-Host ''

    $GitCommand = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $GitCommand) {
        $GitStatus = @(& $GitCommand.Source -C $ProjectDirectory status --porcelain 2>$null)
        if ($GitStatus.Count -gt 0) {
            Write-Warning 'The DAMA working tree already contains local changes. This launcher will leave them untouched.'
            Write-Host ''
        }
    }

    $PathSeparator = [System.IO.Path]::PathSeparator
    $SourceDirectory = Join-Path $ProjectDirectory 'src'
    $ExistingPythonPath = $SavedEnvironment['PYTHONPATH']
    if ([string]::IsNullOrWhiteSpace($ExistingPythonPath)) {
        $env:PYTHONPATH = $SourceDirectory
    } else {
        $env:PYTHONPATH = "$SourceDirectory$PathSeparator$ExistingPythonPath"
    }
    $env:PROCESS_TITLE = 'micro-trainer'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_MAX_THREADS = '1'

    Push-Location -LiteralPath $ProjectDirectory
    $LocationPushed = $true

    Write-Host 'Running Windows Python and CUDA preflight checks...'
    $PreflightCode = @'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: CUDA is unavailable in this Windows environment. "
        f"PyTorch={torch.__version__}, torch CUDA={torch.version.cuda}"
    )

import numpy
import psutil
import yaml
import dama.ai.ml.trainer

device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device)
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"GPU: {properties.name}")
print(f"VRAM: {properties.total_memory / (1024 ** 3):.1f} GB")
print("DAMA trainer import: OK")
'@

    $PreflightArguments = @(
        'run', '--no-capture-output', '-n', $CondaEnvironment,
        'python', '-W', 'ignore::FutureWarning', '-c', $PreflightCode
    )
    Invoke-CondaCommand -Arguments $PreflightArguments
    if ($NativeExitCode -ne 0) {
        Write-Host ''
        Write-Host 'Preflight failed. No training was started.' -ForegroundColor Red
        Write-Host "Install a CUDA-enabled Windows PyTorch build in Conda environment '$CondaEnvironment'."
        Write-Host 'Official selector: https://pytorch.org/get-started/locally/'
        throw "Windows CUDA preflight exited with code $NativeExitCode."
    }

    Write-Host ''
    Write-Host 'Preflight passed.' -ForegroundColor Green
    if ($ValidateOnly) {
        Write-Host 'Validation-only mode requested. Training was not started.'
        return
    }

    $TrainerArguments = @(
        'run', '--no-capture-output', '-n', $CondaEnvironment,
        'python', '-W', 'ignore::FutureWarning',
        '-m', 'dama.ai.ml.trainer',
        '--config', $ConfigPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ResumePath)) {
        $TrainerArguments += @('--resume', $ResumePath)
        Write-Host "Resume policy: checkpoint $ResumePath"
    } elseif ($FreshStart) {
        Write-Host 'Resume policy: fresh start'
    } else {
        $TrainerArguments += '--resume-latest'
        Write-Host 'Resume policy: latest checkpoint'
    }

    if (-not [string]::IsNullOrWhiteSpace($TrainDuration)) {
        $TrainerArguments += @('--train-duration', $TrainDuration)
        Write-Host "Training duration override: $TrainDuration"
    }

    Write-Host ''
    Write-Host 'CUDA verified. Starting native Windows training...'
    Invoke-CondaCommand -Arguments $TrainerArguments
    if ($NativeExitCode -ne 0) {
        throw "DAMA training exited with code $NativeExitCode. See log: $LogFile"
    }

    Write-Host 'DAMA training completed successfully.' -ForegroundColor Green
} finally {
    if ($LocationPushed) {
        Pop-Location
    }

    foreach ($Name in $EnvironmentNames) {
        $PreviousValue = $SavedEnvironment[$Name]
        if ($null -eq $PreviousValue) {
            Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        } else {
            [System.Environment]::SetEnvironmentVariable($Name, $PreviousValue, 'Process')
        }
    }

    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
    if ($MutexAcquired) {
        [void] $Mutex.ReleaseMutex()
    }
    if ($null -ne $Mutex) {
        $Mutex.Dispose()
    }
}
