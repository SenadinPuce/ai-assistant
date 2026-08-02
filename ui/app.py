import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from chat_store import ChatStore

API_URL = "http://127.0.0.1:8000"
DB_PATH = str(Path(__file__).parent.parent / "data" / "chats.db")
API_TIMEOUT_SECONDS = 120
UPLOAD_TIMEOUT_SECONDS = 300
CHAT_HISTORY_WINDOW = 10
SUPPORTED_UPLOAD_TYPES = ("txt", "pdf", "docx", "md", "csv", "json", "html", "htm")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = ChatStore(DB_PATH)

store: ChatStore = st.session_state.store

if "current_chat_id" not in st.session_state:
    chats = store.list_chats()
    if not chats:
        chats = [store.create_chat()]
    st.session_state.current_chat_id = chats[0]["id"]

if "renaming_chat_id" not in st.session_state:
    st.session_state.renaming_chat_id = None

if "deleting_chat_id" not in st.session_state:
    st.session_state.deleting_chat_id = None

if "upload_in_progress" not in st.session_state:
    st.session_state.upload_in_progress = False

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

if "upload_widget_key" not in st.session_state:
    st.session_state.upload_widget_key = 0


def start_upload() -> None:
    """Mark the selected documents for ingestion before Streamlit reruns."""
    st.session_state.upload_in_progress = True


def format_file_size(size_bytes: int) -> str:
    """Return a compact, user-friendly file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def set_upload_result(success: bool, message: str) -> None:
    """Store upload feedback to be shown after the next rerun."""
    st.session_state.upload_result = {"success": success, "message": message}


def validate_uploaded_files(uploaded_files: list[Any]) -> list[str]:
    """Return validation errors for the current file selection."""
    errors: list[str] = []

    unsupported_files = [
        file.name
        for file in uploaded_files
        if Path(file.name).suffix.lower().lstrip(".") not in SUPPORTED_UPLOAD_TYPES
    ]
    empty_files = [file.name for file in uploaded_files if file.size == 0]

    if unsupported_files:
        errors.append(
            "Nepodržani tipovi fajlova: " + ", ".join(sorted(unsupported_files))
        )
    if empty_files:
        errors.append("Prazni fajlovi ne mogu biti dodani: " + ", ".join(sorted(empty_files)))

    return errors


def format_request_error(exc: requests.exceptions.RequestException, fallback: str) -> str:
    """Create a cleaner user-facing error message from a failed HTTP request."""
    response = getattr(exc, "response", None)
    if response is None:
        return f"{fallback}: {exc}"

    try:
        payload = response.json()
    except ValueError:
        payload = None

    detail = payload.get("detail") if isinstance(payload, dict) else response.text.strip()
    if detail:
        return f"{fallback}: {detail}"

    return f"{fallback}: HTTP {response.status_code}"


def upload_documents(files_payload: list[tuple[str, tuple[str, bytes, str]]]) -> dict[str, Any]:
    """Send the selected files to the upload endpoint."""
    response = requests.post(
        f"{API_URL}/documents/upload",
        files=files_payload,
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _parse_sse_stream(response: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse a text/event-stream response body into (event_type, payload) tuples."""
    event_type = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        line = (raw_line or "").rstrip("\r")

        if line == "":
            if data_lines:
                yield event_type, json.loads("\n".join(data_lines))
            event_type, data_lines = "message", []
            continue

        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    if data_lines:
        yield event_type, json.loads("\n".join(data_lines))


def stream_chat_events(
    question: str, history: list[dict[str, str]]
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Send a chat request to the API and yield parsed SSE (event_type, payload) tuples."""
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "history": history},
        timeout=API_TIMEOUT_SECONDS,
        stream=True,
    )
    response.raise_for_status()
    try:
        yield from _parse_sse_stream(response)
    finally:
        response.close()


def list_documents() -> list[dict[str, Any]]:
    """Fetch documents currently ingested into the knowledge base."""
    response = requests.get(f"{API_URL}/documents", timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("documents", [])


def delete_document(document_id: str) -> None:
    """Remove a document and its chunks from the knowledge base."""
    response = requests.delete(f"{API_URL}/documents/{document_id}", timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render numbered sources (matching inline [n] citations) with an excerpt each."""
    for idx, src in enumerate(sources, start=1):
        if src.get("source"):
            label = f"**[{idx}] {src['source']}**"
            if src.get("page"):
                label += f", str. {src['page']}"
        elif src.get("url"):
            label = f"**[{idx}]** [{src['url']}]({src['url']})"
        else:
            continue

        st.markdown(label)
        if src.get("snippet"):
            st.markdown(f"> {src['snippet']}")


def render_chat_message(message: dict[str, Any]) -> None:
    """Render a stored chat message, including any assistant sources."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Izvori"):
                render_sources(message["sources"])


def apply_sidebar_styles() -> None:
    """Apply lightweight sidebar polish while keeping the default Streamlit layout."""
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
                letter-spacing: 0.2px;
            }

            [data-testid="stSidebar"] .stButton > button {
                border-radius: 12px;
                transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.12s ease;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                transform: translateY(-1px);
                border-color: rgba(79, 70, 229, 0.75);
                box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.25);
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                border: 1px solid rgba(79, 70, 229, 0.95);
                box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.35);
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] p {
                font-weight: 600;
            }

            [data-testid="stSidebar"] .stTextInput > div > div > input {
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    apply_sidebar_styles()
    st.title("AI Assistant")

    if st.button("Novi razgovor", icon=":material/add:", use_container_width=True):
        new_chat = store.create_chat()
        st.session_state.current_chat_id = new_chat["id"]
        st.session_state.renaming_chat_id = None
        st.session_state.deleting_chat_id = None
        st.rerun()

    st.subheader("Dokumenti za bazu")
    uploaded_files = st.file_uploader(
        "Odaberi dokumente",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=True,
        disabled=st.session_state.upload_in_progress,
        key=f"document_upload_{st.session_state.upload_widget_key}",
    ) or []
    upload_validation_errors = validate_uploaded_files(uploaded_files)

    file_count = len(uploaded_files)
    total_size = sum(uploaded_file.size for uploaded_file in uploaded_files)
    if file_count:
        file_label = "fajl" if file_count == 1 else "fajla"
        st.caption(
            f"Spremno za dodavanje: {file_count} {file_label} ({format_file_size(total_size)})"
        )

    for validation_error in upload_validation_errors:
        st.warning(validation_error)

    upload_label = "Dodaj fajlove u bazu"
    if file_count == 1:
        upload_label = "Dodaj 1 fajl u bazu"
    elif file_count > 1:
        upload_label = f"Dodaj {file_count} fajla u bazu"

    st.button(
        upload_label,
        use_container_width=True,
        disabled=(
            not file_count
            or st.session_state.upload_in_progress
            or bool(upload_validation_errors)
        ),
        on_click=start_upload,
    )

    if st.session_state.upload_result:
        result = st.session_state.upload_result
        if result["success"]:
            st.success(result["message"])
        else:
            st.error(result["message"])
        st.session_state.upload_result = None

    if st.session_state.upload_in_progress:
        files_payload = [
            ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
            for file in uploaded_files
        ]
        st.info("Dodavanje dokumenata je u toku...")
        try:
            with st.spinner("Fajlovi se šalju i indeksiraju..."):
                data = upload_documents(files_payload)

            uploaded_file_count = len(data.get("files", []))
            uploaded_file_label = "fajl" if uploaded_file_count == 1 else "fajla"
            set_upload_result(
                True,
                (
                    f"Dodano {uploaded_file_count} {uploaded_file_label} i "
                    f"{data.get('chunks', 0)} segmenata u bazu."
                ),
            )
            st.session_state.upload_widget_key += 1
        except requests.exceptions.RequestException as exc:
            set_upload_result(False, format_request_error(exc, "Neuspješan upload"))
        finally:
            st.session_state.upload_in_progress = False
            st.rerun()

    st.divider()
    st.subheader("Dokumenti u bazi")

    if "deleting_document_id" not in st.session_state:
        st.session_state.deleting_document_id = None

    try:
        ingested_documents = list_documents()
    except requests.exceptions.RequestException:
        ingested_documents = []
        st.caption("Nije moguće učitati listu dokumenata.")

    if not ingested_documents:
        st.caption("Baza znanja je trenutno prazna.")

    for document in ingested_documents:
        document_id = document["id"]

        if st.session_state.deleting_document_id == document_id:
            st.warning(f'Ukloniti "{document["original_filename"]}" iz baze?')
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Da", key=f"confirm_delete_doc_{document_id}", use_container_width=True):
                    try:
                        delete_document(document_id)
                    except requests.exceptions.RequestException as exc:
                        st.error(format_request_error(exc, "Brisanje nije uspjelo"))
                    st.session_state.deleting_document_id = None
                    st.rerun()
            with col2:
                if st.button("Ne", key=f"cancel_delete_doc_{document_id}", use_container_width=True):
                    st.session_state.deleting_document_id = None
                    st.rerun()
        else:
            col1, col2 = st.columns([5, 1])
            with col1:
                chunk_label = "segment" if document["chunk_count"] == 1 else "segmenata"
                st.caption(f"{document['original_filename']} · {document['chunk_count']} {chunk_label}")
            with col2:
                if st.button("🗑️", key=f"delete_doc_{document_id}", help="Ukloni ovaj dokument"):
                    st.session_state.deleting_document_id = document_id
                    st.rerun()

    st.divider()

    for chat in store.list_chats():
        chat_id = chat["id"]
        is_active = chat_id == st.session_state.current_chat_id

        if st.session_state.renaming_chat_id == chat_id:
            new_title = st.text_input(
                "Novo ime",
                value=chat["title"],
                key=f"rename_input_{chat_id}",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Potvrdi", key=f"confirm_rename_{chat_id}", use_container_width=True):
                    if new_title.strip():
                        store.rename_chat(chat_id, new_title)
                    st.session_state.renaming_chat_id = None
                    st.rerun()
            with col2:
                if st.button("Odustani", key=f"cancel_rename_{chat_id}", use_container_width=True):
                    st.session_state.renaming_chat_id = None
                    st.rerun()

        elif st.session_state.deleting_chat_id == chat_id:
            st.warning(f'Obrisati "{chat["title"]}"?')
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Da", key=f"confirm_delete_{chat_id}", use_container_width=True):
                    store.delete_chat(chat_id)
                    st.session_state.deleting_chat_id = None
                    remaining = store.list_chats()
                    if not remaining:
                        remaining = [store.create_chat()]
                    st.session_state.current_chat_id = remaining[0]["id"]
                    st.rerun()
            with col2:
                if st.button("Ne", key=f"cancel_delete_{chat_id}", use_container_width=True):
                    st.session_state.deleting_chat_id = None
                    st.rerun()

        else:
            label = f"{chat['title']}" if is_active else chat["title"]
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                if st.button(
                    label,
                    key=f"select_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.renaming_chat_id = None
                    st.session_state.deleting_chat_id = None
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"rename_{chat_id}", help="Rename this chat"):
                    st.session_state.renaming_chat_id = chat_id
                    st.session_state.deleting_chat_id = None
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                    st.session_state.deleting_chat_id = chat_id
                    st.session_state.renaming_chat_id = None
                    st.rerun()

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

current_chat = store.get_chat(st.session_state.current_chat_id)
if current_chat is None:
    # The chat was deleted mid-session; fall back to the first available one.
    chats = store.list_chats()
    if not chats:
        chats = [store.create_chat()]
    st.session_state.current_chat_id = chats[0]["id"]
    current_chat = chats[0]

st.header(current_chat["title"])

messages = store.get_messages(st.session_state.current_chat_id)
if not messages:
    st.info("Postavite prvo pitanje ili dodajte dokumente iz bočne trake da proširite bazu znanja.")

for msg in messages:
    render_chat_message(msg)

prompt = st.chat_input(
    "Postavite pitanje…",
    disabled=st.session_state.upload_in_progress,
)
if prompt:
    store.add_message(st.session_state.current_chat_id, "user", prompt)

    # Auto-title: replace default title with the first user message (truncated).
    if current_chat["title"] == "Novi razgovor":
        auto_title = prompt[:50].rstrip() + ("…" if len(prompt) > 50 else "")
        store.rename_chat(st.session_state.current_chat_id, auto_title)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Obrađujem pitanje…", expanded=True)
        sources_holder: dict[str, list] = {"sources": []}

        def _stream_answer() -> Iterator[str]:
            """Update the live status per CRAG stage while yielding answer tokens."""
            reached_tokens = False
            for event_type, payload in stream_chat_events(prompt, history):
                if event_type == "stage" and payload.get("status") == "started":
                    status.update(label=payload["label"], state="running")
                elif event_type == "token":
                    if not reached_tokens:
                        status.update(label="Odgovor je spreman.", state="complete", expanded=False)
                        reached_tokens = True
                    yield payload["text"]
                elif event_type == "sources":
                    sources_holder["sources"] = payload.get("sources", [])
                elif event_type == "error":
                    status.update(label="Greška prilikom generisanja odgovora.", state="error")
                    raise RuntimeError(payload.get("message", "Nepoznata greška"))

            if not reached_tokens:
                status.update(label="Odgovor je spreman.", state="complete", expanded=False)

        try:
            all_messages = store.get_messages(st.session_state.current_chat_id)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in all_messages[-(CHAT_HISTORY_WINDOW + 1):-1]
            ]

            answer = st.write_stream(_stream_answer())
            sources = sources_holder["sources"]

            if sources:
                with st.expander("Izvori"):
                    render_sources(sources)

            store.add_message(
                st.session_state.current_chat_id,
                "assistant",
                answer,
                sources=sources,
            )

        except requests.exceptions.ConnectionError:
            st.error("Nije moguće povezati se sa API serverom.")
        except requests.exceptions.HTTPError as exc:
            st.error(format_request_error(exc, "API greška"))
        except requests.exceptions.RequestException as exc:
            st.error(format_request_error(exc, "Zahtjev nije uspio"))
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Došlo je do neočekivane greške tokom generisanja odgovora.")
