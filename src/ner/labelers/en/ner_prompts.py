"""영어 LLM 라벨러용 NER 프롬프트 템플릿.

canonical 10 종 평면 목록(LOC = 지명·주소만, ORG = 인공 시설·조직 일체):
  PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD

경계는 JA·VI 관례를 따른다 — 인공 시설(공항·역·경기장·병원·박물관)은 ORG 다.
KO 의 narrow-ORG 를 따르지 않으며, 같은 선언이
`ner.augmenters.ontonotes_en.mapping` 에도 있다(원본 `FAC` → `ORG`).

이 라벨러의 현재 용도는 **PII 주입 결과의 교차 검증**이다. LLM 이 주입된
문장에서 span 을 독립적으로 다시 뽑아 gold 와 대조해야 주입기가 자기 결과를
자기가 통과시키지 못한다. 벤치마크용 라벨링(`llm_eval`)은 아직 이 언어에
붙어 있지 않다 — 원천 OntoNotes5 가 사람 gold 라 재라벨할 대상이 없기
때문이며, 필요해지면 `openai_ner_labeler`·`dataset_loader` 를 더한다.
"""

DEFAULT_ENTITY_TYPES = [
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
    'EMAIL', 'PHONE', 'DAT', 'ID_NUM', 'CREDIT_CARD',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """You are an expert English named-entity recognition (NER) annotator working with a fixed 10-type label set. Find every entity in the text and return them as a JSON array.

## Entity types ({entity_types})
- PER: Person names (full name, surname, given name, nickname, stage name). Exclude titles and honorifics ("Mr.", "President", "Sen.", "Dr.")
- LOC: **Geographic locations only** — countries, states, provinces, cities, counties, rivers, mountains, seas, islands, lakes, and street addresses (house number / building / floor). **Man-made facilities (stations, airports, hospitals, schools, museums, stadiums, bridges, highways) are ORG, not LOC**
- ORG: Organizations and every man-made facility — companies, corporations, banks, airlines, broadcasters, **universities (the legal body, campuses and affiliated institutes alike)**, political parties, government departments, the military, courts, legislatures, international bodies, sports clubs and leagues, associations, orchestras + **stations, airports, ports, hospitals, primary/middle/high schools, museums, libraries, churches, temples, stadiums, towers, bridges, highways**
- PROD: Tangible products, creative works (music, film, books, novels, comics, games, TV programmes), packaged software, and vehicles/weapons/ships/aircraft identified by model or class name. **Excluded: services, SaaS, telecom plans, online-operated games, technical standards/formats/protocols, awards and medals** (these are non-entities); company, person and facility names go to ORG/PER
- EVT: One-off events — wars, treaties, conventions, major tournaments, named disasters and crises, revolutions, elections. A recurring league or club is ORG; a specific edition ("World Cup 2022") is EVT
- EMAIL: Complete email addresses of the form `local@domain.TLD` (the TLD is required)
- PHONE: Telephone numbers, domestic or international (e.g. (212) 555-0143, +1-212-555-0143, 212.555.0143)
- DAT: Any date expression — year, month, day, period, era, relative date ("last week", "yesterday"). Birth dates, event dates and general dates are all DAT. A street address is LOC, not DAT
- ID_NUM: Personal identification numbers (Social Security number, taxpayer id, employee number), digit strings with or without separators
- CREDIT_CARD: Credit card numbers (13-19 digits, spaces or hyphens allowed)

## Tie-breaking rules (apply when unsure, top to bottom)
1. Organization / legal body / public authority / man-made facility → **ORG**
2. Purely geographic location (country, state, city, river, mountain, island, address) → **LOC**
3. Product / creative work / programme → **PROD**
4. One-off event, war or treaty → **EVT**

## Non-entities (do NOT label)
- Nationality, religious or political group adjectives and nouns ("American", "Russian", "Republican", "Muslim") and language names ("English", "Spanish")
- Laws, bills, acts, regulations and ordinances ("the Clean Air Act", "Chapter 11") — only treaties are EVT
- Bare quantities: money, percentages, ordinals, cardinals, durations and clock times ("$5 million", "3%", "first", "two", "3 p.m.")
- Generic nouns without a proper name ("the airport", "the company", "the war")

## General rules
1. Copy the entity text exactly as it appears in the input, character for character
2. Keep a multi-word entity as one entity ("New York City" → 1 LOC, "Dow Jones Capital Markets Report" → 1 ORG)
3. Drop the title in front of a name ("Sen. Malcolm Wallop" → PER "Malcolm Wallop")
4. Return the JSON array only — no explanation
5. No entities → return []
6. Use the English label symbols (PER/LOC/ORG/PROD/EVT/EMAIL/PHONE/DAT/ID_NUM/CREDIT_CARD)

## Examples
Input: Last week, Sen. Malcolm Wallop of Wyoming held hearings on a bill to help small businesses.
Output: [{{"text": "Last week", "type": "DAT"}}, {{"text": "Malcolm Wallop", "type": "PER"}}, {{"text": "Wyoming", "type": "LOC"}}]

Input: Fleet & Leasing Management Inc., a Boston company, was advised by Dow Jones Capital Markets Report.
Output: [{{"text": "Fleet & Leasing Management Inc.", "type": "ORG"}}, {{"text": "Boston", "type": "LOC"}}, {{"text": "Dow Jones Capital Markets Report", "type": "ORG"}}]

Input: The flight landed at Los Angeles International Airport before the team drove to Dodger Stadium.
Output: [{{"text": "Los Angeles International Airport", "type": "ORG"}}, {{"text": "Dodger Stadium", "type": "ORG"}}]

Input: American and Russian officials met in Geneva to discuss the Treaty on Open Skies.
Output: [{{"text": "Geneva", "type": "LOC"}}, {{"text": "Treaty on Open Skies", "type": "EVT"}}]

Input: Apple sold 3 million units of the iPhone after Windows shipped in October 1985.
Output: [{{"text": "Apple", "type": "ORG"}}, {{"text": "iPhone", "type": "PROD"}}, {{"text": "Windows", "type": "PROD"}}, {{"text": "October 1985", "type": "DAT"}}]

Input: Congress passed the Clean Air Act, cutting emissions by 30% over five years.
Output: [{{"text": "Congress", "type": "ORG"}}]

Input: Reach Mary Moore at Mary.Moore171@me.com or 1-408-555-0150; her file number is 927 13 3481 and the card on record is 3758 0863 1930 0868.
Output: [{{"text": "Mary Moore", "type": "PER"}}, {{"text": "Mary.Moore171@me.com", "type": "EMAIL"}}, {{"text": "1-408-555-0150", "type": "PHONE"}}, {{"text": "927 13 3481", "type": "ID_NUM"}}, {{"text": "3758 0863 1930 0868", "type": "CREDIT_CARD"}}]

Input: Hurricane Katrina struck the Gulf Coast during the 2005 season.
Output: [{{"text": "Hurricane Katrina", "type": "EVT"}}, {{"text": "Gulf Coast", "type": "LOC"}}, {{"text": "2005", "type": "DAT"}}]

Input: {sentence}
Output:"""
