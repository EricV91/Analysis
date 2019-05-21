:START
@echo on
cd /d "C:\Program Files\Agisoft\Metashape Pro"
call metashape.exe -r "E:\200 Projects\203 ACT_project\create_HQ.py"
TIMEOUT /T 1800
::GOTO START
