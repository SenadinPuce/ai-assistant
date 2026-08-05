# AI Assistant — Inteligentni istraživački asistent (CRAG)

AI asistent zasnovan na **Corrective Retrieval-Augmented Generation (CRAG)** arhitekturi: kombinuje
domensko znanje pohranjeno u vektorskoj bazi (Pinecone) sa velikim jezičkim modelima (OpenAI), a kada
dohvaćeni kontekst nije dovoljno relevantan, dinamički proširuje znanje pretragom interneta (Tavily).

Ovaj repozitorij je implementacija za završni rad *"Primjena velikih jezičkih modela za razvoj AI
aplikacija: Inteligentni istraživački asistent"*.

Za arhitekturu sistema i tok podataka kroz CRAG graf pogledajte [docs/architecture.md](docs/architecture.md).
Za API ugovor pogledajte [docs/api.md](docs/api.md). Za korištenje korisničkog sučelja pogledajte
[docs/user-guide.md](docs/user-guide.md).

## Preduslovi

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (preporučeno) ili `pip`
- Nalozi i API ključevi za: [OpenAI](https://platform.openai.com/), [Pinecone](https://www.pinecone.io/),
  [Tavily](https://tavily.com/), te opciono [LangSmith](https://smith.langchain.com/) za monitoring.

## Postavljanje

1. Instalirajte zavisnosti:

   ```powershell
   uv sync
   ```

   (ili `pip install -e .` ako ne koristite `uv`)

2. Kopirajte `.env.example` u `.env` i popunite vrijednosti:

   ```powershell
   Copy-Item .env.example .env
   ```

   Obavezne varijable: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `TAVILY_API_KEY`.
   `LANGSMITH_*` varijable su opcione — omogućavaju praćenje/tracing poziva u LangSmith UI-ju.

## Pokretanje

Pokrenite API server i korisničko sučelje u odvojenim terminalima:

```powershell
uv run uvicorn api.main:app --reload
```

```powershell
uv run streamlit run ui/app.py
```

API po defaultu sluša na `http://127.0.0.1:8000`, a Streamlit UI se automatski povezuje na tu adresu
(`API_URL` u [ui/app.py](ui/app.py)).

## Testiranje

```powershell
uv run pytest
```

Testovi koji zahtijevaju prave API ključeve (integracijski testovi za LLM/retrieval pozive) se
automatski preskaču (`skipif`) ako odgovarajući ključ nije postavljen u okruženju.

## Evaluacija kvaliteta odgovora

Ručna evaluacija tačnosti odgovora (odvojeno od pytest paketa jer zahtijeva prave, plaćene API pozive):

```powershell
uv run python -m evaluation.run_eval
```

Skripta učitava/kreira golden Q&A skup ([evaluation/dataset.py](evaluation/dataset.py)) u LangSmith,
pokreće CRAG graf nad svakim pitanjem i ocjenjuje odgovore (LLM-as-judge tačnost + provjera citiranja
izvora). Rezultati se pregledaju u LangSmith UI-ju (link se ispisuje u konzoli nakon završetka).

## Struktura projekta

| Direktorij | Odgovornost |
|---|---|
| `api/` | FastAPI REST sloj — SSE streaming chat endpoint, upravljanje dokumentima |
| `rag/` | CRAG logika i LangGraph orkestracija (čvorovi, lanci, stanje grafa) |
| `retrieval/` | Pinecone vektorska baza — konfiguracija i pretraga |
| `ui/` | Streamlit korisničko sučelje |
| `ingestion.py` | Ekstrakcija teksta, chunkanje i upsertovanje dokumenata u Pinecone |
| `evaluation/` | Golden Q&A skup i LangSmith evaluacijska skripta |
| `tests/` | Pytest test suite |
| `docs/` | Dokumentacija i akademske reference |
