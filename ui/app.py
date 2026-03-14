import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Postavite pitanje…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generiše se odgovor…"):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt},
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

                st.session_state.messages.append(
                    {"role": "assistant", "content": stored}
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Nije moguće povezati se sa API serverom. "
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API greška: {e.response.status_code} — {e.response.text}")
