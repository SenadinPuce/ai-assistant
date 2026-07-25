from pathlib import Path

import requests
import streamlit as st

from chat_store import ChatStore

API_URL = "http://127.0.0.1:8000"
DB_PATH = str(Path(__file__).parent.parent / "data" / "chats.db")

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

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("AI Assistant")

    if st.button("+ Novi razgovor", use_container_width=True):
        new_chat = store.create_chat()
        st.session_state.current_chat_id = new_chat["id"]
        st.session_state.renaming_chat_id = None
        st.session_state.deleting_chat_id = None
        st.rerun()

    st.subheader("Dokumenti za bazu")
    uploaded_files = st.file_uploader(
        "Odaberi dokumente",
        type=["txt", "pdf", "docx", "md", "csv", "json", "html", "htm"],
        accept_multiple_files=True,
        disabled=st.session_state.upload_in_progress,
        key=f"document_upload_{st.session_state.upload_widget_key}",
    ) or []

    file_count = len(uploaded_files)
    total_size = sum(uploaded_file.size for uploaded_file in uploaded_files)
    if file_count:
        file_label = "fajl" if file_count == 1 else "fajla"
        st.caption(
            f"Spremno za dodavanje: {file_count} {file_label} ({format_file_size(total_size)})"
        )

    upload_label = "Dodaj fajlove u bazu"
    if file_count == 1:
        upload_label = "Dodaj 1 fajl u bazu"
    elif file_count > 1:
        upload_label = f"Dodaj {file_count} fajla u bazu"

    st.button(
        upload_label,
        use_container_width=True,
        disabled=not file_count or st.session_state.upload_in_progress,
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
        with st.status("Dodavanje dokumenata u bazu", expanded=True) as status:
            try:
                st.write(f"Obrađuje se {len(files_payload)} fajlova.")
                with st.spinner("Fajlovi se šalju i indeksiraju..."):
                    resp = requests.post(
                        f"{API_URL}/documents/upload",
                        files=files_payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                status.update(label="Dokumenti su uspješno dodati u bazu.", state="complete")
                uploaded_file_count = len(data.get("files", []))
                uploaded_file_label = "fajl" if uploaded_file_count == 1 else "fajla"
                st.session_state.upload_result = {
                    "success": True,
                    "message": (
                        f"Dodano {uploaded_file_count} {uploaded_file_label} i "
                        f"{data.get('chunks', 0)} segmenata u bazu."
                    ),
                }
                st.session_state.upload_widget_key += 1
            except requests.exceptions.RequestException as exc:
                status.update(label="Učitavanje dokumenata nije uspjelo.", state="error")
                st.session_state.upload_result = {
                    "success": False,
                    "message": f"Neuspešan upload: {exc}",
                }
            finally:
                st.session_state.upload_in_progress = False
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
            label = f"**{chat['title']}**" if is_active else chat["title"]
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                if st.button(label, key=f"select_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.renaming_chat_id = None
                    st.session_state.deleting_chat_id = None
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"rename_{chat_id}",help="Rename this chat"):
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
for msg in messages:
    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):

            with st.expander("Izvori"):

                for src in msg["sources"]:

                    if src.get("source"):

                        label = src["source"]

                        if src.get("page"):
                            label += f", str. {src['page']}"

                        st.markdown(f"- {label}")

                    elif src.get("url"):
                        st.markdown(
                            f"- {src['url']}"
                        )

if prompt := st.chat_input("Postavite pitanje…"):
    store.add_message(st.session_state.current_chat_id, "user", prompt)

    # Auto-title: replace default title with the first user message (truncated).
    if current_chat["title"] == "Novi razgovor":
        auto_title = prompt[:50].rstrip() + ("…" if len(prompt) > 50 else "")
        store.rename_chat(st.session_state.current_chat_id, auto_title)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generiše se odgovor…"):
            try:
                all_messages = store.get_messages(st.session_state.current_chat_id)
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in all_messages[-11:-1]
                ]

                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt, "history": history},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Izvori"):
                        for src in sources:
                            if src.get("source"):
                                label = src["source"]
                                if src.get("page"):
                                    label += f", str. {src['page']}"
                                st.markdown(f"- {label}")
                            elif src.get("url"):
                                st.markdown(f"- [{src['url']}]({src['url']})")

              
                store.add_message(st.session_state.current_chat_id, "assistant", answer, sources=sources)

            except requests.exceptions.ConnectionError:
                st.error("Nije moguće povezati se sa API serverom.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API greška: {e.response.status_code} — {e.response.text}")
