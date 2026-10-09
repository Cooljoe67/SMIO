[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$GcsBucket = "",
    [string]$GcsConfigObject = "",
    [string]$AccountId = "",
    [string]$ImapHost = "",
    [string]$ImapUser = "",
    [string]$Email = "",
    [string]$FolderList = "default"
)

# Stores a new IMAP account's password in Secret Manager and prints the entry
# to add to the service's config in GCS. The service reads the password at
# login, so the account needs no redeploy.

$ErrorActionPreference = "Stop"

$DotEnvValues = @{}
$DotEnvPath = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $DotEnvPath) {
    foreach ($line in Get-Content $DotEnvPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $key = $Matches[1]
            $value = $Matches[2].Trim()
            if ($value.Length -ge 2 -and
                (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                    ($value.StartsWith("'") -and $value.EndsWith("'")))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $DotEnvValues[$key] = $value
        }
    }
}

function Resolve-ConfigValue {
    param(
        [string]$ParameterValue,
        [string]$Name,
        [hashtable]$FileValues
    )

    if (-not [string]::IsNullOrWhiteSpace($ParameterValue)) {
        return $ParameterValue
    }
    $environmentValue = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) {
        return $environmentValue
    }
    if ($FileValues.ContainsKey($Name)) {
        return [string]$FileValues[$Name]
    }
    return ""
}

$ProjectId = Resolve-ConfigValue $ProjectId "GCP_PROJECT_ID" $DotEnvValues
$GcsBucket = Resolve-ConfigValue $GcsBucket "GCS_BUCKET" $DotEnvValues
$GcsConfigObject = Resolve-ConfigValue $GcsConfigObject "GCS_CONFIG_OBJECT" $DotEnvValues
if ([string]::IsNullOrWhiteSpace($GcsConfigObject)) {
    $GcsConfigObject = "config/smio.yaml"
}

if ([string]::IsNullOrWhiteSpace($ProjectId) -or
    [string]::IsNullOrWhiteSpace($GcsBucket)) {
    throw "Set GCP_PROJECT_ID and GCS_BUCKET in .env or pass -ProjectId and -GcsBucket."
}

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Invoke-Gcloud {
    param([string[]]$Arguments)

    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed with exit code ${LASTEXITCODE}: gcloud $($Arguments -join ' ')"
    }
}

function Test-GcloudSecret {
    param([string]$Name)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & gcloud secrets describe $Name "--project=$ProjectId" "--format=value(name)" 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Set-GcloudSecret {
    # Creates the secret, or adds a version: the service reads the latest one.
    param([string]$Name, [string]$Value)

    # Read from a file: piping the value to gcloud would add a newline to the secret.
    $valueFile = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllText($valueFile, $Value)
        if (Test-GcloudSecret $Name) {
            Invoke-Gcloud @(
                "secrets", "versions", "add", $Name,
                "--data-file=$valueFile",
                "--project=$ProjectId"
            ) | Out-Null
            Write-Host "Added a new version of secret '$Name'."
        }
        else {
            Invoke-Gcloud @(
                "secrets", "create", $Name,
                "--replication-policy=automatic",
                "--data-file=$valueFile",
                "--project=$ProjectId"
            ) | Out-Null
            Write-Host "Created secret '$Name'."
        }
    }
    finally {
        Remove-Item -LiteralPath $valueFile -Force -ErrorAction SilentlyContinue
    }
}

function Read-HiddenValue {
    param([string]$Prompt)

    $secureValue = Read-Host -Prompt $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Read-RequiredValue {
    param([string]$Value, [string]$Prompt)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        $Value = Read-Host -Prompt $Prompt
    }
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "A value for '$Prompt' is required."
    }
    return $Value.Trim()
}

function ConvertTo-YamlString {
    param([string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

Assert-Command "gcloud"

$AccountId = Read-RequiredValue $AccountId "Account id (lowercase letters, digits, hyphens)"
# Part of the secret name, and the key of the account's stored mails.
if ($AccountId -cnotmatch '^[a-z0-9][a-z0-9-]*$') {
    throw "Account id '$AccountId' may only contain lowercase letters, digits and hyphens."
}
$ImapHost = Read-RequiredValue $ImapHost "IMAP host"
$ImapUser = Read-RequiredValue $ImapUser "IMAP username"
$password = Read-HiddenValue "IMAP password"
if ([string]::IsNullOrEmpty($password)) {
    throw "An IMAP password is required."
}

$SecretName = "imap-$AccountId-password"
Set-GcloudSecret $SecretName $password

$GcsUri = "gs://$GcsBucket/$GcsConfigObject"
$LocalCopy = Join-Path ([System.IO.Path]::GetTempPath()) "smio.yaml"
if ([string]::IsNullOrWhiteSpace($Email)) {
    $Email = $ImapUser
}
$YamlHost = ConvertTo-YamlString $ImapHost
$YamlUser = ConvertTo-YamlString $ImapUser
$YamlEmail = ConvertTo-YamlString $Email

Write-Host ""
Write-Host "Add this entry under 'accounts:' in the service's config (adjust the mailbox"
Write-Host "folder names to this server):"
Write-Host ""
Write-Host @"
  - id: $AccountId
    provider: imap
    email: $YamlEmail
    folder_list: $FolderList
    mailbox:
      inbox: INBOX
      folder_root: INBOX
      trash: Papierkorb
    imap:
      host: $YamlHost
      username: $YamlUser
      password_secret: $SecretName
"@
Write-Host ""
Write-Host "Download the config, add the entry, and upload it again:"
Write-Host "  gcloud storage cp `"$GcsUri`" `"$LocalCopy`""
Write-Host "  gcloud storage cp `"$LocalCopy`" `"$GcsUri`""
Write-Host ""
Write-Host "The next workflow run picks it up. Never change the id afterwards: the"
Write-Host "account's stored mails are keyed by it."
