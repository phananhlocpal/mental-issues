param(
    [string]$PythonTag = "3.10",
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

Step "Checking Python launcher"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.10+ first."
}

Step "Creating virtual environment at '$VenvDir' (if needed)"
if (-not (Test-Path $VenvDir)) {
    py -$PythonTag -m venv $VenvDir
}

$PyExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
    throw "Python executable not found in virtual environment: $PyExe"
}

Step "Upgrading pip tooling"
& $PyExe -m pip install --upgrade pip setuptools wheel

Step "Installing PyTorch CUDA build (cu128)"
& $PyExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Step "Installing PyG core"
& $PyExe -m pip install torch-geometric==2.7.0

Step "Installing optional PyG CUDA extensions"
try {
    & $PyExe -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
} catch {
    Warn "Could not install some optional PyG extensions. Continuing with available packages."
}

Step "Installing project requirements"
& $PyExe -m pip install -r requirements.txt

Step "Running smoke test"
& $PyExe -c "import torch, torch_geometric; import transformers, datasets, sklearn, pandas, matplotlib, seaborn, nltk, scipy, yaml, tqdm, networkx; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); x=torch.randn(256,256, device='cuda') if torch.cuda.is_available() else torch.randn(256,256); y=torch.randn(256,256, device=x.device); z=(x@y).mean(); print('matmul ok:', float(z)); print('pyg:', torch_geometric.__version__)"

Write-Host "`nSetup completed successfully." -ForegroundColor Green
Write-Host "Activate with: .\\.venv\\Scripts\\Activate.ps1"
