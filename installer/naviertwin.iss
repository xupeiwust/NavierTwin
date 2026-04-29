; NavierTwin Inno Setup 스크립트 (Windows)
; 사용: `iscc installer\naviertwin.iss` (Inno Setup 6.x)
; PyInstaller 로 `dist/NavierTwin/` 를 먼저 빌드한 뒤 이 스크립트를 돌린다.

[Setup]
AppName=NavierTwin
AppVersion=4.2.58
AppPublisher=NavierTwin Contributors
AppPublisherURL=https://github.com/younglin90/NavierTwin
DefaultDirName={autopf}\NavierTwin
DefaultGroupName=NavierTwin
OutputBaseFilename=NavierTwinSetup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
SetupIconFile=
UninstallDisplayIcon={app}\NavierTwin.exe
DisableProgramGroupPage=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; PyInstaller onedir 출력
Source: "..\dist\NavierTwin\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 샘플 데이터 & 아이콘 (있다면)
Source: "..\resources\*"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\NavierTwin"; Filename: "{app}\NavierTwin.exe"
Name: "{group}\{cm:UninstallProgram,NavierTwin}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\NavierTwin"; Filename: "{app}\NavierTwin.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\NavierTwin.exe"; Description: "{cm:LaunchProgram,NavierTwin}"; Flags: nowait postinstall skipifsilent
