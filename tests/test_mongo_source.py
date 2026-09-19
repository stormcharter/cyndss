from datetime import datetime, timezone

from bson import ObjectId

from shipper.models import SourceType
from shipper.mongo_source import MongoSource


def test_to_raw_event_reads_nested_source_metadata():
    event = MongoSource._to_raw_event(
        {
            "_id": ObjectId(),
            "source": {
                "source_type": "website",
                "channel_name": "Example feed",
                "message_id": "article-1",
                "url": "https://example.test/article-1",
            },
            "text": "Threat report",
            "collected_at": datetime.now(timezone.utc),
        }
    )

    assert event.source.source_type is SourceType.WEBSITE
    assert event.source.channel_name == "Example feed"
    assert event.source.message_id == "article-1"
    assert event.source.url == "https://example.test/article-1"


def test_to_raw_event_accepts_authoritative_corpus_sources():
    event = MongoSource._to_raw_event(
        {
            "_id": ObjectId(),
            "source_type": "authoritative",
            "channel_name": "cisa_kev",
            "message_id": "CVE-2026-0001",
            "text": "Known exploited vulnerability from the CISA KEV catalog",
            "collected_at": datetime.now(timezone.utc),
        }
    )

    assert event.source.source_type is SourceType.AUTHORITATIVE


def test_to_raw_event_keeps_legacy_flat_source_metadata():
    event = MongoSource._to_raw_event(
        {
            "_id": ObjectId(),
            "source_type": "telegram",
            "channel_name": "Legacy channel",
            "message_id": 42,
            "text": "Legacy threat report",
            "collected_at": datetime.now(timezone.utc),
        }
    )

    assert event.source.source_type is SourceType.TELEGRAM
    assert event.source.channel_name == "Legacy channel"
    assert event.source.message_id == "42"
