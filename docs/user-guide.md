# Korisnički vodič

Kratak vodič za korištenje Streamlit korisničkog sučelja ([ui/app.py](../ui/app.py)).

## Razgovor sa asistentom

1. Unesite pitanje u polje na dnu ekrana i pritisnite Enter.
2. Dok asistent obrađuje pitanje, prikazuje se živi status napretka (npr. "Pretražujem bazu
   znanja…", "Rangiram rezultate po relevantnosti…", "Tražim dodatne informacije na webu…") —
   ovo odražava korake CRAG toka rada opisane u [docs/architecture.md](architecture.md).
3. Odgovor se generiše token po token uživo. Ako je odgovor zasnovan na dokumentima ili web
   pretrazi, sadrži brojčane citate poput `[1]`, `[2]`.
4. Kliknite na "Izvori" ispod odgovora da vidite na koje dokumente/URL-ove citati upućuju,
   zajedno sa kratkim isječkom teksta.

## Upravljanje razgovorima

- **Novi razgovor** — dugme u bočnoj traci započinje prazan razgovor (čuva se tek nakon prve poruke).
- Svaki razgovor u listi može se preimenovati (✏️) ili obrisati (🗑️) klikom na odgovarajuću ikonu.
- Naziv razgovora se automatski postavlja prema prvoj poslanoj poruci.

## Dodavanje dokumenata u bazu znanja

1. U sekciji "Dokumenti za bazu" odaberite jedan ili više fajlova (podržani formati: PDF, DOCX,
   TXT, MD, CSV, JSON, HTML).
2. Kliknite "Dodaj fajl(ove) u bazu". Fajlovi se šalju, ekstrahuje se tekst, dijele na segmente
   (chunkove) i pohranjuju u Pinecone vektorsku bazu.
3. Po završetku, poruka potvrđuje broj dodanih fajlova i segmenata.

## Pregled i brisanje dokumenata

- Sekcija "Dokumenti u bazi" prikazuje sve trenutno ingestovane dokumente sa brojem segmenata.
- Klikom na 🗑️ pored dokumenta i potvrdom, dokument i svi njegovi segmenti trajno se uklanjaju
  iz baze znanja.

## Napomene

- Dok se dokument dodaje ili se odgovor generiše, unos novih pitanja i navigacija između
  razgovora su privremeno onemogućeni kako bi se izbjegli konflikti.
- Historija razgovora se lokalno čuva u SQLite bazi (`data/chats.db`) i preživljava restart
  aplikacije.
- Informacije o ingestovanim dokumentima (uključujući broj chunk-ova i        Pinecone vector ID-jeve)
se čuvaju u `data/documents.db`; raw upload fajlovi se ne čuvaju trajno nakon obrade.
