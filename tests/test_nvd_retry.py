"""
Юніт-тести NVDClient.lookup: ретраї на ТИМЧАСОВИХ збоях NVD (429/5xx/мережа).

Баг: lookup робив рівно одну спробу; один 503 від NVD -> UNREACHABLE, і реальна
CVE (напр. CVE-2021-44228) мовчки йшла в індекс неверифікованою.

Мережі немає: httpx.MockTransport + лічильник викликів; sleep підмінено.

Запуск:
    pytest tests/test_nvd_retry.py -v
або:
    python tests/test_nvd_retry.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Settings читає .env з cwd (там ключі поза схемою) -> на час імпорту йдемо
# в порожній каталог (див. tests/test_verifier_revoked.py).
_cwd_before_import = os.getcwd()
with tempfile.TemporaryDirectory() as _tmpdir:
    os.chdir(_tmpdir)
    try:
        import httpx  # noqa: E402

        import shipper.verifier as verifier_mod  # noqa: E402
        from shipper.config import settings  # noqa: E402
        from shipper.verifier import NVDClient, VerificationStatus  # noqa: E402
    finally:
        os.chdir(_cwd_before_import)

CVE = "CVE-2021-44228"
OK_BODY = {
    "vulnerabilities": [
        {
            "cve": {
                "descriptions": [{"lang": "en", "value": "Log4Shell"}],
                "metrics": {
                    "cvssMetricV31": [{"cvssData": {"baseScore": 10.0}}]
                },
            }
        }
    ]
}


def _run(responses, *, sleeps=None, lookups=1):
    """Ганяє lookup(CVE) `lookups` разів; responses — послідовність відповідей
    (httpx.Response або виняток). Повертає (результати, лічильник, sleeps, client)."""
    calls = {"n": 0}
    sleeps = [] if sleeps is None else sleeps

    def handler(request):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        item = responses[i]
        if isinstance(item, Exception):
            raise item
        return item

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            client = NVDClient(c)
            return [await client.lookup(CVE) for _ in range(lookups)], client

    orig = verifier_mod._sleep
    verifier_mod._sleep = fake_sleep
    try:
        results, client = asyncio.run(go())
    finally:
        verifier_mod._sleep = orig
    return results, calls["n"], sleeps, client


def _ok():
    return httpx.Response(200, json=OK_BODY)


def test_503_503_200_verified_after_three_calls():
    results, calls, sleeps, _ = _run(
        [httpx.Response(503), httpx.Response(503), _ok()]
    )
    assert results[0].status == VerificationStatus.VERIFIED
    assert calls == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0], "backoff має бути експоненційним"


def test_429_retry_after_is_respected():
    results, calls, sleeps, _ = _run(
        [httpx.Response(429, headers={"Retry-After": "7"}), _ok()]
    )
    assert results[0].status == VerificationStatus.VERIFIED
    assert calls == 2
    assert sleeps == [7.0]


def test_retry_after_is_capped():
    _, _, sleeps, _ = _run(
        [httpx.Response(503, headers={"Retry-After": "3600"}), _ok()]
    )
    assert sleeps == [settings.nvd_retry_after_max_seconds]


def test_404_is_not_retried():
    results, calls, sleeps, _ = _run([httpx.Response(404)])
    assert results[0].status == VerificationStatus.NOT_FOUND
    assert calls == 1
    assert sleeps == []


def test_403_is_not_retried():
    results, calls, sleeps, _ = _run([httpx.Response(403)])
    assert results[0].status == VerificationStatus.UNREACHABLE
    assert calls == 1
    assert sleeps == []


def test_persistent_503_exhausts_attempts_and_is_not_cached():
    results, calls, _, _ = _run([httpx.Response(503)], lookups=2)
    assert all(r.status == VerificationStatus.UNREACHABLE for r in results)
    assert calls == 2 * settings.nvd_max_retries, (
        "UNREACHABLE не кешується: другий lookup знову йде в мережу"
    )


def test_transport_error_then_200_verified():
    results, calls, sleeps, _ = _run(
        [httpx.ConnectTimeout("timeout"), _ok()]
    )
    assert results[0].status == VerificationStatus.VERIFIED
    assert calls == 2
    assert len(sleeps) == 1


def test_sleep_happens_outside_semaphore():
    """Під час backoff слот семафора вільний для інших CVE."""
    held = []

    def handler(request):
        return httpx.Response(503)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            client = NVDClient(c)

            async def fake_sleep(seconds):
                held.append(client._sem._value)  # вільні слоти

            verifier_mod._sleep = fake_sleep
            await client.lookup(CVE)
            return client

    orig = verifier_mod._sleep
    try:
        client = asyncio.run(go())
    finally:
        verifier_mod._sleep = orig
    full = 5 if not settings.nvd_api_key else 20
    assert held and all(v == full for v in held), "семафор тримався під час сну"


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
