"""
Конфігурація шипера через змінні оточення.

Усі секрети (API-ключі NVD) і URL беруться з .env, а не хардкодяться.
"""

from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # MongoDB
    mongo_uri: str = Field(
        default="mongodb://localhost:27017/?replicaSet=rs0",
        validation_alias=AliasChoices("SHIPPER_MONGO_URI", "MONGO_URI"),
    )
    mongo_db: str = "threat_intel"
    mongo_collection: str = "raw_events"

    # Ембединги. EMBED_PROVIDER приймає ollama, remote або auto (спершу remote,
    # потім Ollama). Для індексації та пошуку має бути ОДИН провайдер: різні
    # моделі дають несумісні векторні простори, навіть якщо обидві повертають 768 вимірів.
    embed_provider: str = "ollama"
    embed_dims: int = 768
    remote_embed_url: str = ""
    remote_embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    remote_embed_api_key: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # LLM — будь-який OpenAI-сумісний ендпойнт (OpenRouter / Lambda / локальна
    # Ollama через /v1). За замовчуванням хмара: на сервері без GPU генерація
    # на CPU надто повільна (~30-40с/пост). Локально:
    # LLM_BASE_URL=http://localhost:11434/v1, LLM_MODEL=hermes3.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str | None = None
    llm_model: str = "nousresearch/hermes-3-llama-3.1-405b:free"

    # Bearer по відкритому HTTP заборонений — окрім loopback. Docker-бридж
    # (`http://ollama:11434/v1`) формально не loopback, хоча назовні не світить;
    # для нього є цей ЯВНИЙ опт-аут. Мовчазного послаблення перевірки немає:
    # хто вмикає — той свідомо бере ризик на себе.
    allow_plaintext_llm_key: bool = False

    # --- Заземлення міток (grounding) ---------------------------------
    # Модель сама вирішує, які CVE/TTP навісити, а верифікація підтверджує лише
    # ІСНУВАННЯ ідентифікатора. Тому перед верифікацією стоїть гейт:
    #   "literal" — лише дослівна згадка ідентифікатора в тексті (ДЕФОЛТ);
    #   "quoted"  — плюс мітки, підкріплені цитатою, яка справді є в документі;
    #   "off"     — стара поведінка, приймати все (аварійний важіль).
    # Дефолт саме "literal", а не мʼякший "quoted": A/B на бойовому корпусі
    # (hermes3:8b, 07.09.2026) показав, що з вимогою цитувати модель перестає
    # штампувати вироджений T1190, але починає навішувати ІНШІ техніки,
    # підкріплюючи їх справжніми, але не по темі цитатами. Тобто "quoted" міняє
    # форму вигадки, а не прибирає її. Цитата лишається корисною як сигнал, але
    # достатнім доказом сама по собі не є.
    # Literal-тип, а не str: помилка в .env впаде на старті, а не тихо вимкне гейт.
    grounding_mode: Literal["literal", "quoted", "off"] = "literal"

    # Manticore
    manticore_url: str = "http://localhost:9308"  # HTTP-ендпойнт
    manticore_table: str = "events_enriched"


    # NVD API (опційно — без ключа ліміт 5 запитів / 30 с, із ключем 50)
    nvd_api_key: str | None = None
    nvd_base_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    # MITRE ATT&CK — STIX-бандл завантажуємо при старті й кешуємо
    mitre_stix_url: str = (
        "https://raw.githubusercontent.com/mitre/cti/master/"
        "enterprise-attack/enterprise-attack.json"
    )

    # Дедуплікація
    dedup_similarity_threshold: float = 0.92  # косинусна схожість
    dedup_window_size: int = 1000  # скільки останніх ембедингів тримати в RAM

    # Пакетна вставка
    batch_size: int = 50
    batch_timeout_seconds: float = 2.0

    # Надійність Manticore: ретраї та dead-letter колекція
    manticore_max_retries: int = 3
    manticore_retry_base_seconds: float = 0.5
    dead_letter_collection: str = "shipper_dead_letters"

    # Надійне підтвердження запису (durable ack): черга до сінка та ретраї конвеєра
    sink_queue_maxsize: int = 200
    pipeline_max_retries: int = 2
    pipeline_retry_base_seconds: float = 1.0
    pipeline_concurrency: int = 32

    # Оренда CDC, перепідключення, межі ресурсів і loopback-ендпойнт готовності
    shipper_lease_seconds: float = 60.0
    shipper_stale_seconds: float = 120.0
    shipper_reconnect_base_seconds: float = 1.0
    shipper_reconnect_max_seconds: float = 30.0
    shipper_http_timeout_seconds: float = 60.0
    shipper_shutdown_timeout_seconds: float = 20.0
    shipper_max_event_text_chars: int = 1_000_000
    shipper_health_port: int = 9100
    fast_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("SHIPPER_FAST_MODE"),
    )
    @field_validator("embed_provider", mode="before")
    @classmethod
    def normalize_embed_provider(cls, value: object) -> str:
        provider = str(value).strip().lower()
        if provider not in {"ollama", "remote", "auto"}:
            raise ValueError("EMBED_PROVIDER must be ollama, remote, or auto")
        return provider

    @model_validator(mode="after")
    def resolve_and_validate(self) -> "Settings":
        self.llm_base_url = self.llm_base_url.strip()
        self.llm_model = self.llm_model.strip()

        self._validate_url(
            self.mongo_uri,
            "SHIPPER_MONGO_URI",
            {"mongodb", "mongodb+srv"},
        )
        self._validate_url(self.manticore_url, "MANTICORE_URL", {"http", "https"})
        self._validate_url(self.llm_base_url, "LLM_BASE_URL", {"http", "https"})
        self._validate_url(self.ollama_url, "OLLAMA_URL", {"http", "https"})
        if self.embed_provider == "remote" and not self.remote_embed_url.strip():
            raise ValueError("REMOTE_EMBED_URL is required for the remote provider")
        if self.remote_embed_url:
            self._validate_url(
                self.remote_embed_url,
                "REMOTE_EMBED_URL",
                {"http", "https"},
            )

        if not self.allow_plaintext_llm_key:
            self._require_tls_for_secret(
                self.llm_base_url,
                self.llm_api_key,
                "LLM_BASE_URL",
            )
        self._require_tls_for_secret(
            self.remote_embed_url,
            self.remote_embed_api_key,
            "REMOTE_EMBED_URL",
        )

        positive_integers = {
            "MANTICORE_MAX_RETRIES": self.manticore_max_retries,
            "PIPELINE_MAX_RETRIES": self.pipeline_max_retries,
            "PIPELINE_CONCURRENCY": self.pipeline_concurrency,
            "SINK_QUEUE_MAXSIZE": self.sink_queue_maxsize,
            "BATCH_SIZE": self.batch_size,
            "DEDUP_WINDOW_SIZE": self.dedup_window_size,
            "EMBED_DIMS": self.embed_dims,
            "SHIPPER_MAX_EVENT_TEXT_CHARS": self.shipper_max_event_text_chars,
        }
        invalid_integers = [
            name for name, value in positive_integers.items() if value < 1
        ]
        if invalid_integers:
            raise ValueError(f"{', '.join(invalid_integers)} must be positive")

        positive_durations = {
            "BATCH_TIMEOUT_SECONDS": self.batch_timeout_seconds,
            "MANTICORE_RETRY_BASE_SECONDS": self.manticore_retry_base_seconds,
            "PIPELINE_RETRY_BASE_SECONDS": self.pipeline_retry_base_seconds,
            "SHIPPER_RECONNECT_BASE_SECONDS": self.shipper_reconnect_base_seconds,
            "SHIPPER_HTTP_TIMEOUT_SECONDS": self.shipper_http_timeout_seconds,
            "SHIPPER_SHUTDOWN_TIMEOUT_SECONDS": self.shipper_shutdown_timeout_seconds,
        }
        invalid_durations = [
            name for name, value in positive_durations.items() if value <= 0
        ]
        if invalid_durations:
            raise ValueError(f"{', '.join(invalid_durations)} must be positive")

        if not 0.0 < self.dedup_similarity_threshold <= 1.0:
            raise ValueError("DEDUP_SIMILARITY_THRESHOLD must be in (0, 1]")
        if self.shipper_lease_seconds < 3.0:
            raise ValueError("SHIPPER_LEASE_SECONDS must be at least 3 seconds")
        if self.shipper_stale_seconds < self.shipper_lease_seconds:
            raise ValueError(
                "SHIPPER_STALE_SECONDS must be >= SHIPPER_LEASE_SECONDS"
            )
        if (
            self.shipper_reconnect_max_seconds
            < self.shipper_reconnect_base_seconds
        ):
            raise ValueError("Shipper reconnect bounds are invalid")
        if not 1 <= self.shipper_health_port <= 65535:
            raise ValueError("SHIPPER_HEALTH_PORT must be between 1 and 65535")

        required_names = {
            "MONGO_DB": self.mongo_db,
            "MONGO_COLLECTION": self.mongo_collection,
            "DEAD_LETTER_COLLECTION": self.dead_letter_collection,
            "MANTICORE_TABLE": self.manticore_table,
            "LLM_MODEL": self.llm_model,
            "OLLAMA_EMBED_MODEL": self.ollama_embed_model,
        }
        empty_names = [
            name for name, value in required_names.items() if not value.strip()
        ]
        if empty_names:
            raise ValueError(f"{', '.join(empty_names)} must not be empty")
        return self

    @staticmethod
    def _validate_url(value: str, name: str, allowed_schemes: set[str]) -> None:
        parsed = urlparse(value)
        if parsed.scheme not in allowed_schemes or not parsed.netloc:
            choices = ", ".join(sorted(allowed_schemes))
            raise ValueError(
                f"{name} must be an absolute URL with one of these schemes: {choices}"
            )

    @staticmethod
    def _require_tls_for_secret(url: str, secret: str | None, name: str) -> None:
        if not secret or not url:
            return
        parsed = urlparse(url)
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and parsed.hostname not in loopback_hosts:
            raise ValueError(f"{name} must use HTTPS when an API key is configured")


# Валідація налаштувань розвʼязує дефолти провайдерів ще до того, як будь-який
# сервіс імпортує синглтон, — тож обійти перевірки пізнішою мутацією неможливо.
settings = Settings()

