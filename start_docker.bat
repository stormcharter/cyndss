@echo off
chcp 65001 >nul
title Threat Intelligence Launcher
echo [=========================================]
echo [+] Запуск платформи Threat Intelligence [+]
echo [=========================================]
echo.

:: Перехід у теку проєкту (там, де лежить цей файл)
cd /d "%~dp0"

:: Запуск інфраструктури
echo [1/4] Пiднiмаємо Docker-контейнери...
docker compose up -d --build
echo.

:: Пауза для ініціалізації Оллами
timeout /t 5 /nobreak >nul

:: Перевірка та автоматичне завантаження моделей в Ollama
echo [2/4] Перевiрка наявностi моделей в Ollama...
echo Очікуйте, перевіряємо hermes3 та nomic-embed-text...
docker exec diplom-ollama-1 ollama run hermes3 "say ok" >nul 2>&1
if %errorlevel% neq 0 (
    echo [-] Модель hermes3 не знайдена. Завантажуємо з мережi...
    docker exec -it diplom-ollama-1 ollama pull hermes3
) else (
    echo [+] Модель hermes3 готова до роботи.
)

docker exec diplom-ollama-1 ollama run nomic-embed-text "say ok" >nul 2>&1
if %errorlevel% neq 0 (
    echo [-] Модель nomic-embed-text не знайдена. Завантажуємо...
    docker exec -it diplom-ollama-1 ollama pull nomic-embed-text
) else (
    echo [+] Модель nomic-embed-text готова до роботи.
)
echo.

:: Відкриття окремого вікна з логами шиппера
echo [3/4] Вiдкриваємо логи шиппера...
start "Shipper Logs" cmd /k "docker compose logs -f shipper"

:: Запуск юзербота
echo [4/4] Запускаємо Telegram-колектор...
start "Telegram Collector" cmd /k "python tg.py"

echo.
echo [=========================================]
echo [+] Всi компоненти успiшно запущенi!      [+]
echo [=========================================]
echo.
pause