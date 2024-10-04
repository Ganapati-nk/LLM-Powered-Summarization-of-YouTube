import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader,UnstructuredURLLoader



st.set_page_config(
    page_title="Summarize text from youtube",
    page_icon=":rocket:",  
)

st.title("LangChain: Summarize Text from youtube or Website")
st.subheader("Summarize URL")



import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

llm=ChatGroq(api_key=groq_api_key,model="Gemma-7b-It")
prompt_template=""" 
Provide the detailed summary of the following content in 300 words:
content:{text}
"""


prompt=PromptTemplate(template=prompt_template,input_variables=['text'])

url=st.text_input("URL",label_visibility="collapsed")


if st.button("Summerize"):
    if not url.strip():
        st.error("Enter the URL to Proceed")
    elif not validators.url(url):
        st.error("Enter the Valid URL")

    else:
        try:
            with st.spinner("Waiting....."):
                if "youtu.be" in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=False)
                docs=loader.load()
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")



