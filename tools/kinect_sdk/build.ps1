# Compile KinectGrab.cs avec le csc.exe de .NET Framework (aucun Visual Studio requis).
# Prérequis : Kinect SDK 1.8 (variable KINECTSDK10_DIR).
$ErrorActionPreference = 'Stop'
$sdk = $env:KINECTSDK10_DIR
if (-not $sdk) { throw "Kinect SDK 1.8 introuvable (KINECTSDK10_DIR non défini)." }
$dll = Join-Path $sdk 'Assemblies\Microsoft.Kinect.dll'
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$here = $PSScriptRoot
& $csc /nologo /platform:x64 /optimize+ /out:"$here\kinect_grab.exe" /reference:"$dll" "$here\KinectGrab.cs"
if ($LASTEXITCODE -ne 0) { throw "Compilation échouée" }
Copy-Item $dll $here -Force   # la DLL doit se trouver à côté de l'exe
Write-Output "OK : $here\kinect_grab.exe"
