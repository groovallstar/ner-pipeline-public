"""영어 LLM 라벨러용 NER 프롬프트 템플릿.

canonical 10 종 평면 목록(LOC = 지명·주소·경로, ORG = 조직·개별 구조물):
  PER LOC ORG PROD EVT DAT EMAIL PHONE ID_NUM CREDIT_CARD

인프라 경계는 canonical §3 을 따른다 — **개별 구조물은 `ORG`, 여러 지점을
잇는 경로는 `LOC`** 다. KO 의 narrow-ORG(인공 시설을 아예 버림)를 따르지
않으며, 같은 선언이 canonical §4.3 에도 있다 — 그쪽은 원본 `FAC` 를
표면별로 두 라벨로 가른 판정표(`docs/manual/data/en-fac-verdicts.json`)를
정본으로 둔다. **선언이 갈리면 검증이 gold 와 다른 자로 재게 되므로**
`tests/ner/labelers/test_en_labeler.py` 가 이 세 자리(이 docstring · LOC 줄 ·
ORG 줄)를 문구로 걸어 둔다.

이 라벨러의 현재 용도는 **PII 주입 결과의 교차 검증**이다. LLM 이 주입된
문장에서 span 을 독립적으로 다시 뽑아 gold 와 대조해야 주입기가 자기 결과를
자기가 통과시키지 못한다. 벤치마크용 라벨링(`llm_eval`)은 아직 이 언어에
붙어 있지 않다 — 원천 OntoNotes5 가 사람 gold 라 재라벨할 대상이 없기
때문이며, 필요해지면 ja·vi 처럼 `dataset_loader` 를 더한다.
"""

DEFAULT_ENTITY_TYPES = [
    'PER', 'LOC', 'ORG', 'PROD', 'EVT',
    'EMAIL', 'PHONE', 'DAT', 'ID_NUM', 'CREDIT_CARD',
]

# ── Single-sentence prompt ────────────────────────────────────────────

SINGLE_PROMPT_TEMPLATE = """You are an expert English named-entity recognition (NER) annotator working with a fixed 10-type label set. Find every entity in the text and return them as a JSON array.

## Entity types ({entity_types})
- PER: Person names (full name, surname, given name, nickname, stage name). Exclude titles and honorifics ("Mr.", "President", "Sen.", "Dr.")
- LOC: **Geographic locations and routes** — countries, states, provinces, cities, counties, rivers, mountains, seas, islands, lakes, street addresses (house number / building / floor), and **routes that connect places — roads, streets, avenues, highways, railway and subway lines**. **A single man-made structure (station, airport, hospital, school, museum, stadium, bridge, tunnel) is ORG, not LOC**
- ORG: Organizations and single man-made structures — companies, corporations, banks, airlines, broadcasters, **universities (the legal body, campuses and affiliated institutes alike)**, political parties, government departments, the military, courts, legislatures, international bodies, sports clubs and leagues, associations, orchestras + **stations, airports, ports, hospitals, primary/middle/high schools, museums, libraries, churches, temples, stadiums, towers, bridges, tunnels**. **A route that connects places (road, highway, railway or subway line) is LOC, not ORG**
- PROD: Tangible products, creative works (music, film, books, novels, comics, games, TV programmes), packaged software, and vehicles/weapons/ships/aircraft identified by model or class name. **Excluded: services, SaaS, telecom plans, online-operated games, technical standards/formats/protocols, awards and medals** (these are non-entities); company, person and facility names go to ORG/PER
- EVT: One-off events — wars, treaties, conventions, major tournaments, named disasters and crises, revolutions, elections. A recurring league or club is ORG; a specific edition ("World Cup 2022") is EVT
- EMAIL: Complete email addresses of the form `local@domain.TLD` (the TLD is required)
- PHONE: Telephone numbers, domestic or international (e.g. (212) 555-0143, +1-212-555-0143, 212.555.0143)
- DAT: Any date expression — year, month, day, period, era, relative date ("last week", "yesterday"). Birth dates, event dates and general dates are all DAT. A street address is LOC, not DAT
- ID_NUM: Personal identification numbers (Social Security number, taxpayer id, employee number), digit strings with or without separators
- CREDIT_CARD: Credit card numbers (13-19 digits, spaces or hyphens allowed)

## Tie-breaking rules (apply when unsure, top to bottom)
1. Organization / legal body / public authority / single man-made structure → **ORG**
2. Geographic location (country, state, city, river, mountain, island, address) or a route that connects places (road, highway, railway or subway line) → **LOC**
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

Input: Traffic on Interstate 95 backed up near the Golden Gate Bridge after the Red Line stopped running.
Output: [{{"text": "Interstate 95", "type": "LOC"}}, {{"text": "Golden Gate Bridge", "type": "ORG"}}, {{"text": "Red Line", "type": "LOC"}}]

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
