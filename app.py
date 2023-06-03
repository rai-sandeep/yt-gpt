import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from llama_index import download_loader

# Setting page title and header
st.set_page_config(page_title="YouTubeGPT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>YouTube Video based chatbot</h1>", unsafe_allow_html=True)

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0
if "video_disabled" not in st.session_state:
    st.session_state["video_disabled"] = False
if "video_link" not in st.session_state:
    st.session_state["video_link"] = ""
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
        k=6
    )

memory = init_memory()

# Reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    st.session_state['video_disabled'] = False
    st.session_state['video_link'] = ""

def prepare(video_link):
    YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    documents = YoutubeTranscriptReader().load_data(ytlinks=[video_link])

    documents = [doc.to_langchain_format() for doc in documents]
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(documents, embeddings).as_retriever(k=4)

    llm = ChatOpenAI(temperature=1, model_name=model, max_tokens=2048)

    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        # verbose=True,
        memory=memory,
        max_tokens_limit=1536
    )
    st.session_state["conversation"] = conversation

def video_disable():
    st.session_state["video_disabled"] = True

# Generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    conversation = st.session_state['conversation']
    completion = conversation({'question': prompt})
    st.session_state['conversation'] = conversation
    
    response = completion['answer']
    st.session_state['messages'].append({"role": "assistant", "content": response})

    return response

video_input = st.text_input("Enter the YouTube link: ", key='video', 
                            placeholder=st.session_state.video_link, 
                            disabled=st.session_state.video_disabled, 
                            on_change=video_disable)

if video_input:
    prepare(video_input)

# Container for chat history
response_container = st.container()
# Container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("You:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
