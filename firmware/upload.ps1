# Compile and upload the arm firmware to an Arduino Nano.
#   .\upload.ps1 COM5            (find the port in Device Manager -> Ports)
# Tries the "old bootloader" (most Nano clones) first, then the new one.
param([Parameter(Mandatory = $true)][string]$Port)

$sketch = Join-Path $PSScriptRoot "arm"
arduino-cli core install arduino:avr | Out-Null
arduino-cli lib install Servo | Out-Null
foreach ($cpu in "atmega328old", "atmega328") {
    Write-Host "trying Nano ($cpu) on $Port ..."
    arduino-cli compile --fqbn "arduino:avr:nano:cpu=$cpu" $sketch
    if ($LASTEXITCODE -ne 0) { Write-Host "compile failed: the sketch or the Servo library is the problem, not the port"; exit 1 }
    arduino-cli upload -p $Port --fqbn "arduino:avr:nano:cpu=$cpu" $sketch
    if ($LASTEXITCODE -eq 0) { Write-Host "uploaded OK ($cpu)"; exit 0 }
}
Write-Host "upload failed: check the port and the USB cable (must be a data cable)"
exit 1
