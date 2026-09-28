[CmdletBinding()]
param(
    [string]$ProjectId = "smio-509409",
    [string]$Region = "europe-west3",
    [string]$ServiceName = "smio",
    [string]$Repository = "smio",
    [string]$Cpu = "2",
    [string]$Memory = "4Gi",
    [int]$SummaryStartHour = 8,
    [string]$SummaryTimezone = "Europe/Berlin",
    [string]$GcsBucket = "smio-marcus-my-gcp-project-artifacts",
    [string]$GcsModelPrefix = "models/distilbert_deployed",
    [string]$GcsDatabaseObject = "databases/smio.db",
    [string]$RuntimeServiceAccount = "",
    [string]$ImapHostSecret = "imap-host",
    [string]$ImapUserSecret = "imap-user",
    [string]$ImapPasswordSecret = "imap-password",
    [switch]$IncludeGmailSecrets,
    [string]$GmailClientIdSecret = "gmail-client-id",
    [string]$GmailClientSecretSecret = "gmail-client-secret",
    [string]$GmailRefreshTokenSecret = "gmail-refresh-token",
    [string]$GmailToAddressSecret = "gmail-to-address"
)

$ErrorActionPreference = "Stop"

function Invoke-Gcloud {
    param([string[]]$Arguments)

    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed with exit code ${LASTEXITCODE}: gcloud $($Arguments -join ' ')"
    }
}

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Wait-ForDocker {
    $startupTimeoutSeconds = 180
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    while ($true) {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            docker info *> $null
            $dockerExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($dockerExitCode -eq 0) {
            return
        }

        if ($timer.Elapsed.TotalSeconds -ge $startupTimeoutSeconds) {
            throw "Docker Desktop did not become ready within $startupTimeoutSeconds seconds. Start it manually and rerun deploy.ps1."
        }

        Write-Host "Waiting for Docker Desktop to finish starting..."
        Start-Sleep -Seconds 5
    }
}

function Start-DockerDesktop {
    if (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue) {
        return
    }

    $desktopPath = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $desktopPath)) {
        $desktopPath = Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker\Docker Desktop.exe"
    }
    if (-not (Test-Path $desktopPath)) {
        throw "Docker Desktop was not found in the standard install locations. Start it manually and rerun deploy.ps1."
    }

    Write-Host "Docker Desktop is not running. Starting it..."
    Start-Process -FilePath $desktopPath
}

Assert-Command "docker"
Assert-Command "gcloud"

$ActiveGcloudAccount = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not check Google Cloud authentication status."
}
if ([string]::IsNullOrWhiteSpace($ActiveGcloudAccount)) {
    Write-Host "No active Google Cloud login found. Starting Google Cloud login..."
    Invoke-Gcloud @("auth", "login")
}
else {
    Write-Host "Google Cloud SDK is already authenticated. Skipping login."
}

Invoke-Gcloud @("config", "set", "project", $ProjectId)

Start-DockerDesktop
Wait-ForDocker

if ([string]::IsNullOrWhiteSpace($GcsBucket) -or $GcsBucket -eq "YOUR_BUCKET") {
    throw "A real GCS bucket is required. Current value: '$GcsBucket'"
}

$Image = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ServiceName`:latest"
$ArtifactRegistryHost = "$Region-docker.pkg.dev"

Write-Host "Project:    $ProjectId"
Write-Host "Region:     $Region"
Write-Host "Image:      $Image"
Write-Host "GCS bucket: $GcsBucket"

Invoke-Gcloud @("services", "enable", "artifactregistry.googleapis.com", "run.googleapis.com", "secretmanager.googleapis.com")

$repositoryCheckErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$repositoryOutput = & gcloud artifacts repositories describe $Repository "--location=$Region" "--project=$ProjectId" 2>&1
$repositoryExitCode = $LASTEXITCODE
$ErrorActionPreference = $repositoryCheckErrorAction

$repositoryExists = $false
$repositoryExists = ($repositoryExitCode -eq 0)

if (-not $repositoryExists) {
    Invoke-Gcloud @(
        "artifacts", "repositories", "create", $Repository,
        "--repository-format=docker", "--location=$Region", "--project=$ProjectId"
    )
}

Invoke-Gcloud @("auth", "configure-docker", $ArtifactRegistryHost, "--quiet")

Write-Host "Building Docker image..."
docker build -t $Image .
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with exit code $LASTEXITCODE"
}

Write-Host "Pushing Docker image..."
docker push $Image
if ($LASTEXITCODE -ne 0) {
    throw "Docker push failed with exit code $LASTEXITCODE"
}

if ([string]::IsNullOrWhiteSpace($RuntimeServiceAccount)) {
    $projectNumber = (& gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
    $RuntimeServiceAccount = "$projectNumber-compute@developer.gserviceaccount.com"
}

Write-Host "Granting runtime access to Secret Manager and GCS..."
Invoke-Gcloud @(
    "projects", "add-iam-policy-binding", $ProjectId,
    "--member=serviceAccount:$RuntimeServiceAccount",
    "--role=roles/secretmanager.secretAccessor",
    "--condition=None"
)
Invoke-Gcloud @(
    "storage", "buckets", "add-iam-policy-binding", "gs://$GcsBucket",
    "--member=serviceAccount:$RuntimeServiceAccount",
    "--role=roles/storage.objectAdmin"
)

$secretBindings = @(
    "IMAP_HOST=$ImapHostSecret`:latest",
    "IMAP_USER=$ImapUserSecret`:latest",
    "IMAP_PASSWORD=$ImapPasswordSecret`:latest"
)

if ($IncludeGmailSecrets) {
    $secretBindings += "GMAIL_CLIENT_ID=$GmailClientIdSecret`:latest"
    $secretBindings += "GMAIL_CLIENT_SECRET=$GmailClientSecretSecret`:latest"
    $secretBindings += "GMAIL_REFRESH_TOKEN=$GmailRefreshTokenSecret`:latest"
    $secretBindings += "GMAIL_TO_ADDRESS=$GmailToAddressSecret`:latest"
}

Write-Host "Deploying Cloud Run service..."
$deployArguments = @(
    "run", "deploy", $ServiceName,
    "--image=$Image",
    "--project=$ProjectId",
    "--region=$Region",
    "--port=8080",
    "--no-allow-unauthenticated",
    "--max-instances=1",
    "--concurrency=1",
    "--timeout=3600",
    "--cpu=$Cpu",
    "--memory=$Memory",
    "--service-account=$RuntimeServiceAccount",
    "--set-env-vars=GCS_BUCKET=$GcsBucket,GCS_CLASSIFIER_MODEL_PREFIX=$GcsModelPrefix,GCS_DATABASE_OBJECT=$GcsDatabaseObject,SMIO_SUMMARY_START_HOUR=$SummaryStartHour,SMIO_SUMMARY_TIMEZONE=$SummaryTimezone",
    "--set-secrets=$($secretBindings -join ',')"
)
Invoke-Gcloud $deployArguments

Write-Host "Deployment completed successfully."
$ServiceUrl = (& gcloud run services describe $ServiceName --project=$ProjectId --region=$Region --format="value(status.url)").Trim()
Write-Host "Service URL: $ServiceUrl"
Write-Host "Smoke test: `$token = gcloud auth print-identity-token; Invoke-RestMethod '$ServiceUrl/' -Headers @{ Authorization = ``"Bearer `$token``" }"