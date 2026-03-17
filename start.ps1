param(
    [int]$Port = 8010,
    [switch]$SkipChecks,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "[start] $Message" -ForegroundColor Cyan
}

function Test-PortAvailable {
    param([int]$LocalPort)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $LocalPort)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener -ne $null) {
            $listener.Stop()
        }
    }
}

function Get-AvailablePort {
    param([int]$PreferredPort)
    if (Test-PortAvailable -LocalPort $PreferredPort) {
        return $PreferredPort
    }

    for ($candidate = $PreferredPort + 1; $candidate -le $PreferredPort + 50; $candidate++) {
        if (Test-PortAvailable -LocalPort $candidate) {
            return $candidate
        }
    }

    throw "No free port found between $PreferredPort and $($PreferredPort + 50)."
}

if (-not (Test-Path ".env")) {
    throw "Missing .env file. Create it from .env.example before starting the API."
}

if (-not $SkipChecks) {
    Write-Step "Validating environment variables..."
    & python scripts/verify_setup.py
    if ($LASTEXITCODE -ne 0) {
        throw "Environment validation failed."
    }
}

$SelectedPort = Get-AvailablePort -PreferredPort $Port
if ($SelectedPort -ne $Port) {
    Write-Step "Port $Port is busy. Using $SelectedPort instead."
}
else {
    Write-Step "Using port $SelectedPort."
}

$UvicornArgs = @(
    "-m", "uvicorn",
    "api.main:app",
    "--host", "0.0.0.0",
    "--port", "$SelectedPort"
)

if (-not $NoReload) {
    $UvicornArgs += "--reload"
}

Write-Step "API URL: http://localhost:$SelectedPort/"
Write-Step "Starting FastAPI server..."
& python @UvicornArgs
