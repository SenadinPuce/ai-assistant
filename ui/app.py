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

                stored = answer
                if sources:
                    stored += "\n\n**Izvori:**\n"
                    for src in sources:
                        if src.get("source"):
                            line = src["source"]
                            if src.get("page"):
                                line += f", str. {src['page']}"
                            stored += f"- {line}\n"
                        elif src.get("url"):
                            stored += f"- [{src['url']}]({src['url']})\n"

                store.add_message(st.session_state.current_chat_id, "assistant", stored)

            except requests.exceptions.ConnectionError:
                st.error("Nije moguće povezati se sa API serverom.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API greška: {e.response.status_code} — {e.response.text}")
