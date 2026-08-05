# API referenca

Bazni URL: `http://127.0.0.1:8000` (podesivo preko `API_URL` u [ui/app.py](../ui/app.py)).

## `GET /health`

Provjera dostupnosti servisa.

```json
{ "status": "ok" }
```

## `POST /chat`

Pokreće CRAG tok rada za jedno pitanje i vraća `text/event-stream` (Server-Sent Events).

**Zahtjev:**

```json
{
  "question": "Sta su poslovni sistemi?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

**SSE eventi** (u redoslijedu pojavljivanja):

| Event | Payload | Opis |
|---|---|---|
| `stage` | `{ "node": str, "label": str, "status": "started" \| "done" }` | Napredak kroz CRAG čvorove ([STAGE_LABELS](../api/main.py) u api/main.py), za live prikaz u UI-ju |
| `token` | `{ "text": str }` | Komad generisanog odgovora — emituje se isključivo dok se izvršava `GENERATE_ANSWER` ili `DIRECT_ANSWER` čvor |
| `sources` | `{ "sources": [...] }` | Numerisani izvori (uparuju se sa `[1]`, `[2]` citatima u odgovoru); svaki izvor ima `source`/`page` (dokument) ili `url` (web) i `snippet` |
| `done` | `{}` | Kraj streama |
| `error` | `{ "message": str }` | Greška tokom izvršavanja grafa |

## `GET /documents`

Vraća listu dokumenata trenutno ingestovanih u bazu znanja.

```json
{
  "documents": [
    {
      "id": "uuid",
      "original_filename": "primjer.pdf",
      "chunk_count": 12,
      "created_at": "..."
    }
  ]
}
```

## `POST /documents/upload`

Multipart upload jednog ili više fajlova (`files` polje). Podržani formati: `.txt`, `.md`, `.pdf`,
`.docx`, `.csv`, `.json`, `.html`, `.htm`.

**Odgovor:**

```json
{
  "message": "Documents ingested successfully",
  "files": ["primjer.pdf"],
  "chunks": 12
}
```

Greške: `400` za nepodržan tip fajla ili prazan zahtjev.

## `DELETE /documents/{document_id}`

Uklanja dokument i njegove vektore iz Pinecone indeksa te briše zapis iz registra.

**Odgovor:**

```json
{ "message": "Document deleted", "id": "uuid" }
```

Greške: `404` ako dokument ne postoji.
