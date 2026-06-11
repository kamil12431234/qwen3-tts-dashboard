# Qwen3-TTS Dashboard

Web-интерфейс для синтеза речи на основе Qwen3-TTS (voice clone / custom speaker / voice design).

## Быстрый старт

1. **Создать окружение:**
   - `python -m venv venv`
   - `source venv/bin/activate`
   - `pip install -r requirements.txt`
2. **Установить бэкенд TTS:**
   - `pip install faster_qwen3_tts` (или из исходников)
3. **Запустить панель:**
   - `bash start.sh`

Открыть в браузере: http://localhost:8000

## Настройка

Редактировать параметры в `start.sh`:
- `MODEL_PATH` — путь к модели или ID на HuggingFace (по умолчанию Qwen/Qwen3-TTS-1.7B)
- `PORT` — порт веб-интерфейса
- `DEVICE`, `DTYPE`, `ATTN`

## Режимы

- **Voice Clone** — клонирование голоса по эталону (ref_audio + ref_text).
- **Custom Voice** — выбор встроенного спикера.
- **Voice Design** — генерация голоса по текстовому описанию (требует VoiceDesign-модель).

## API

- `GET /api/health` — статус сервиса.
- `POST /api/generate` — синтез речи.
- `GET /api/speakers` — список доступных спикеров.
- `GET /api/outputs` — список сгенерированных файлов.
- `GET /docs` — Swagger-документация.
