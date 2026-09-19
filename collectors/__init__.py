"""
Пакетні колектори даних для платформи CyberIntel.

На відміну від tg.py (живий слухач Telegram у реальному часі), ці колектори
запускаються на вимогу й проходять по списку джерел із channels.json:

    youtube  — текстові транскрипти відео (collectors/youtube.py)
    website  — статті та RSS/Atom-стрічки (collectors/website.py)

Усі вони складають сирі документи в ту саму колекцію raw_events, звідки їх
забирає shipper. Формат документа єдиний (див. collectors/base.py), тож
shipper обробляє youtube/website так само, як telegram: ембеддинг,
дедуплікація, витяг сутностей через Hermes 3, верифікація CVE/MITRE.
"""
