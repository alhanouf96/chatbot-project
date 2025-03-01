import streamlit as st
import requests

st.title("Chatbot basic")

#FastApi take playload then sendit to openAI
chat_url = "http://127.0.0.1:8000/chat/"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
 #st.session_state.messages تضاف لها المحادثه   
#role assistant or end user
#content answer or question
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#chat_message function in stremlit
#markdown

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

#payload josn format
#dict > list > dict
        payload = {
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        } # chat history
        #when send payload use headers
        #notify API  what kind of payload that we will send =  json
        headers = {
            "Content-Type": "application/json"
        }

        # No Stream approach
        #ارسال السؤال
        stream = requests.post(chat_url, json=payload, headers=headers)
         #طلب الرد فقط
        response = stream.json()["reply"]
        #عرضه في markdown
        st.markdown(response)

        # Stream approach
        # def get_stream_response():
        #     with requests.post(chat_url, json=payload, headers=headers, stream=True) as r:
        #         for chunk in r:
        #             yield chunk.decode('utf-8')
        # response = st.write_stream(get_stream_response)

    # حتى يعرض كل الحادثه مع الردود والاسئله
    st.session_state.messages.append({"role": "assistant", "content": response})