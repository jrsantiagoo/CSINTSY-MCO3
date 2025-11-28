@echo off

echo Running squiddyboi 10 times...
for /l %%i in (1,1,10) do (
    echo Trial %%i: squiddyboi
    call py bot.py --cat squiddyboi --render -1
    echo.
)
