FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# prompt_guard спільний з UI (ізоляція недовіреного тексту в промптах)
COPY prompt_guard.py .
# lemmatizer спільний з UI (укр. лематизація content на індексації = у запиті)
COPY lemmatizer.py .
COPY shipper ./shipper

# Непривілейований користувач: shipper нічого на диск не пише (кеш MITRE/NVD —
# в пам'яті), тож root у контейнері не потрібен.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

CMD ["python", "-m", "shipper.main"]
