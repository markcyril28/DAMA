# HOW TO RUN (PowerShell, from the DAMA repository)
#   Preview changes:          .\setup_conda.ps1 -WhatIf
#   Create or update:         .\setup_conda.ps1
#   Validate without changes: .\setup_conda.ps1 -ValidateOnly
#   Recreate explicitly:      .\setup_conda.ps1 -Recreate -Confirm
#   One-process policy bypass:
#     powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_conda.ps1 -ValidateOnly
#
# This script changes only the selected Conda environment, not repository files.
#
#Requires -Version 5.1

<#
.SYNOPSIS
Creates or updates the native Windows Conda environment used by DAMA.

.DESCRIPTION
This script manages only the selected Conda environment. It does not edit the
DAMA repository, build extensions in the source tree, or change Git state.
An existing environment is updated in place unless -Recreate is supplied.
Packages that already import successfully at the requested versions are kept.

Use -ValidateOnly to inspect an existing environment without changing it.
Use -WhatIf to preview environment operations.

.PARAMETER EnvironmentName
Name of the Windows Conda environment. The default is "dama".

.PARAMETER PythonVersion
Python version for the environment. The default is "3.11".

.PARAMETER CudaWheel
PyTorch CUDA wheel channel. The default is "cu128", suitable for the local
RTX 5050 and matching the Linux setup script.

.PARAMETER Recreate
Remove and recreate the named environment. Without this option, an existing
environment is updated in place.

.PARAMETER ValidateOnly
Verify Python, DAMA imports, PyTorch, and CUDA without installing packages.

.EXAMPLE
.\setup_conda.ps1 -ValidateOnly

.EXAMPLE
.\setup_conda.ps1

.EXAMPLE
.\setup_conda.ps1 -Recreate -Confirm
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [ValidatePattern('^[A-Za-z0-9_.-]+$')]
    [string] $EnvironmentName = 'dama',

    [ValidateSet('3.11', '3.12')]
    [string] $PythonVersion = '3.11',

    [ValidateSet('cu128', 'cu130')]
    [string] $CudaWheel = 'cu128',

    [switch] $Recreate,

    [switch] $ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw 'This setup script must run in Windows PowerShell or PowerShell on Windows.'
}

if ($ValidateOnly -and $Recreate) {
    throw 'Use either -ValidateOnly or -Recreate, not both.'
}

$ProjectDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$CondaExecutable = $null
if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE) -and
    (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    $CondaExecutable = [System.IO.Path]::GetFullPath($env:CONDA_EXE)
} else {
    # An activated Conda PowerShell session defines a function named "conda".
    # Resolve only native commands so arguments such as "env list --json" are
    # passed to Conda instead of PowerShell parameter binding.
    $CondaCommand = Get-Command -Name conda.exe, conda.bat `
        -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $CondaCommand) {
        $CondaExecutable = $CondaCommand.Source
    }
}
if ([string]::IsNullOrWhiteSpace($CondaExecutable)) {
    throw 'A native Conda executable was not found. Install Miniconda or open its PowerShell prompt and try again.'
}

$LogRoot = if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    Join-Path $env:LOCALAPPDATA 'DAMA\logs'
} else {
    Join-Path ([System.IO.Path]::GetTempPath()) 'DAMA\logs'
}
$LogTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogRoot "setup_conda_$LogTimestamp.txt"

$Mutex = New-Object System.Threading.Mutex($false, 'Local\DAMA_Conda_Setup_Windows')
$MutexAcquired = $false
$TranscriptStarted = $false
$script:NativeExitCode = 0

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Executable,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $SavedPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can convert native stderr into PowerShell errors.
        # The native process exit code is the authoritative result here.
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments
        $script:NativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $SavedPreference
    }
}

function Invoke-Conda {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments,

        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    Write-Host $Description
    Invoke-NativeCommand -Executable $script:CondaExecutable -Arguments $Arguments
    if ($script:NativeExitCode -ne 0) {
        throw "$Description failed with exit code $script:NativeExitCode."
    }
}

function Test-CondaEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name
    )

    $SavedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $JsonText = & $script:CondaExecutable env list --json 2>$null
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $SavedPreference
    }
    if ($ExitCode -ne 0) {
        throw "Could not query Conda environments. Exit code: $ExitCode"
    }

    $EnvironmentData = $JsonText | ConvertFrom-Json
    foreach ($EnvironmentPath in $EnvironmentData.envs) {
        if ((Split-Path -Leaf $EnvironmentPath) -ieq $Name) {
            return $true
        }
    }
    return $false
}

function Test-EnvironmentPythonCode {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Code
    )

    $Payload = [System.Convert]::ToBase64String(
        [System.Text.Encoding]::UTF8.GetBytes($Code)
    )
    $Bootstrap = "import base64;exec(base64.b64decode('$Payload'))"
    Invoke-NativeCommand -Executable $script:CondaExecutable -Arguments @(
        'run', '--no-capture-output', '-n', $script:EnvironmentName,
        'python', '-W', 'ignore::FutureWarning', '-c', $Bootstrap
    )
    return ($script:NativeExitCode -eq 0)
}

function Invoke-EnvironmentVerification {
    Write-Host 'Verifying Windows Python, project imports, PyTorch, and CUDA...'
    $VerificationCode = @'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}")

try:
    import numpy
    import psutil
    import yaml
    import dama.ai.ml.trainer
except Exception as exc:
    raise SystemExit(f"ERROR: DAMA dependency or trainer import failed: {exc}")

print(f"Python: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to Windows PyTorch.")

device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device)
print(f"GPU: {properties.name}")
print(f"Compute capability: {properties.major}.{properties.minor}")
print(f"VRAM: {properties.total_memory / (1024 ** 3):.1f} GB")
print("DAMA trainer import: OK")
'@
    # conda.bat can truncate literal multiline arguments on Windows. Encode the
    # verification so Python receives it through one ASCII command-line argument.
    $VerificationPayload = [System.Convert]::ToBase64String(
        [System.Text.Encoding]::UTF8.GetBytes($VerificationCode)
    )
    $VerificationBootstrap = "import base64;exec(base64.b64decode('$VerificationPayload'))"

    $SourceDirectory = Join-Path $script:ProjectDirectory 'src'
    $SavedPythonPath = [System.Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    try {
        $PathSeparator = [System.IO.Path]::PathSeparator
        if ([string]::IsNullOrWhiteSpace($SavedPythonPath)) {
            $env:PYTHONPATH = $SourceDirectory
        } else {
            $env:PYTHONPATH = "$SourceDirectory$PathSeparator$SavedPythonPath"
        }

        Invoke-Conda -Description 'Environment verification' -Arguments @(
            'run', '--no-capture-output', '-n', $script:EnvironmentName,
            'python', '-W', 'ignore::FutureWarning', '-c', $VerificationBootstrap
        )
    } finally {
        if ($null -eq $SavedPythonPath) {
            Remove-Item -LiteralPath 'Env:PYTHONPATH' -ErrorAction SilentlyContinue
        } else {
            [System.Environment]::SetEnvironmentVariable('PYTHONPATH', $SavedPythonPath, 'Process')
        }
    }
}

try {
    try {
        $MutexAcquired = $Mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $MutexAcquired = $true
    }
    if (-not $MutexAcquired) {
        throw 'Another DAMA Windows Conda setup process is already running.'
    }

    if (-not $WhatIfPreference) {
        [void] [System.IO.Directory]::CreateDirectory($LogRoot)
        Start-Transcript -Path $LogFile -Append | Out-Null
        $TranscriptStarted = $true
    }

    Write-Host '=== DAMA - Native Windows Conda Setup ==='
    Write-Host "Project:       $ProjectDirectory"
    Write-Host "Environment:   $EnvironmentName"
    Write-Host "Python:        $PythonVersion"
    Write-Host "PyTorch index: $CudaWheel"
    Write-Host "Log:           $LogFile"
    Write-Host ''
    Write-Host 'Repository safeguard: this script does not edit or build inside the DAMA repository.'
    Write-Host ''

    $NvidiaCommand = Get-Command nvidia-smi -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $NvidiaCommand) {
        Write-Warning 'nvidia-smi was not found on PATH. CUDA verification may fail until the Windows NVIDIA driver is available.'
    } else {
        Invoke-NativeCommand -Executable $NvidiaCommand.Source -Arguments @('--query-gpu=name,driver_version', '--format=csv,noheader')
        if ($NativeExitCode -ne 0) {
            Write-Warning "nvidia-smi exited with code $NativeExitCode."
        }
    }
    Write-Host ''

    $EnvironmentExists = Test-CondaEnvironment -Name $EnvironmentName
    Write-Host "Environment exists: $EnvironmentExists"

    if ($ValidateOnly) {
        if (-not $EnvironmentExists) {
            throw "Conda environment '$EnvironmentName' does not exist."
        }
        Invoke-EnvironmentVerification
        Write-Host ''
        Write-Host 'Validation completed successfully. No packages were changed.' -ForegroundColor Green
        return
    }

    if ($Recreate -and $EnvironmentExists) {
        if ($PSCmdlet.ShouldProcess("Conda environment '$EnvironmentName'", 'Remove and recreate')) {
            Invoke-Conda -Description "Removing Conda environment '$EnvironmentName'" -Arguments @(
                'env', 'remove', '-n', $EnvironmentName, '-y'
            )
            $EnvironmentExists = $false
        }
    }

    if (-not $EnvironmentExists) {
        if ($PSCmdlet.ShouldProcess("Conda environment '$EnvironmentName'", "Create with Python $PythonVersion")) {
            Invoke-Conda -Description "Creating Conda environment '$EnvironmentName'" -Arguments @(
                'create', '-n', $EnvironmentName, "python=$PythonVersion", 'pip', 'numpy', '-y'
            )
            $EnvironmentExists = $true
        }
    }

    if ($EnvironmentExists) {
        $CoreCheckCode = @"
import sys

if sys.version_info[:2] != tuple(int(part) for part in "$PythonVersion".split(".")):
    raise SystemExit(1)

try:
    import pip
    import numpy
except Exception:
    raise SystemExit(1)
"@
        if (Test-EnvironmentPythonCode -Code $CoreCheckCode) {
            Write-Host 'Python, pip, and NumPy already satisfy the setup requirements. Skipping core package installation.' -ForegroundColor Green
        } elseif ($PSCmdlet.ShouldProcess("Conda environment '$EnvironmentName'", 'Ensure Python, pip, and NumPy')) {
            Invoke-Conda -Description "Updating core packages in '$EnvironmentName'" -Arguments @(
                'install', '-n', $EnvironmentName, "python=$PythonVersion", 'pip', 'numpy', '-y'
            )
        }
    }

    if (-not $EnvironmentExists) {
        Write-Host 'No environment changes were performed.'
        return
    }

    $DependencyCheckCode = @"
import importlib

modules = (
    "PyQt6", "yaml", "psutil", "pytest", "setproctitle",
    "plotly", "matplotlib", "Cython", "pynvml",
)
try:
    for module in modules:
        importlib.import_module(module)
except Exception:
    raise SystemExit(1)
"@
    if (Test-EnvironmentPythonCode -Code $DependencyCheckCode) {
        Write-Host 'All DAMA Python dependencies are already installed. Skipping dependency installation.' -ForegroundColor Green
    } elseif ($PSCmdlet.ShouldProcess("Conda environment '$EnvironmentName'", 'Install missing DAMA Python dependencies')) {
        Invoke-Conda -Description 'Installing DAMA Python dependencies' -Arguments @(
            'run', '--no-capture-output', '-n', $EnvironmentName,
            'python', '-m', 'pip', 'install',
            'PyQt6', 'pyyaml', 'psutil', 'pytest', 'setproctitle',
            'plotly', 'matplotlib', 'cython', 'nvidia-ml-py'
        )
    }

    $TorchIndexUrl = "https://download.pytorch.org/whl/$CudaWheel"
    $TorchCheckCode = @"
try:
    import torch
    import torchvision
except Exception:
    raise SystemExit(1)

if "+$CudaWheel" not in torch.__version__ or not torch.cuda.is_available():
    raise SystemExit(1)
"@
    if (Test-EnvironmentPythonCode -Code $TorchCheckCode) {
        Write-Host "CUDA-enabled PyTorch ($CudaWheel) is already installed and working. Skipping PyTorch installation." -ForegroundColor Green
    } elseif ($PSCmdlet.ShouldProcess("Conda environment '$EnvironmentName'", "Install or repair CUDA PyTorch from $TorchIndexUrl")) {
        Invoke-Conda -Description "Installing CUDA-enabled PyTorch ($CudaWheel)" -Arguments @(
            'run', '--no-capture-output', '-n', $EnvironmentName,
            'python', '-m', 'pip', 'install', '--upgrade', '--force-reinstall', '--no-cache-dir',
            'torch', 'torchvision', '--index-url', $TorchIndexUrl
        )
    }

    if ($WhatIfPreference) {
        Write-Host 'WhatIf completed. Verification was skipped because no changes were applied.'
        return
    }

    Invoke-EnvironmentVerification
    Write-Host ''
    Write-Host '=== Setup Complete ===' -ForegroundColor Green
    Write-Host "Validate later: .\setup_conda.ps1 -EnvironmentName $EnvironmentName -ValidateOnly"
    Write-Host "Start training: .\local_train.ps1 -CondaEnvironment $EnvironmentName"
    Write-Host 'Cython extensions were intentionally not built in the repository.'
} finally {
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
