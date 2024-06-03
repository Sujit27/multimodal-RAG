import os
import openai
import io
import uuid
import base64
from base64 import b64decode
import numpy as np
from PIL import Image
import time

from unstructured.partition.pdf import partition_pdf

from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain import hub

from operator import itemgetter

import config
from api_key import API_key
from utils import doc_partition, data_category, tables_summarize, encode_image, image_captioning, split_image_text_types, prompt_func


os.environ["OPENAI_API_KEY"] = API_key
openai.api_key = os.environ["OPENAI_API_KEY"]

poppler_path = config.poppler_path
path = config.path
file_name = config.file_name



def main():

    raw_pdf_elements = doc_partition(path,file_name)
    texts = data_category(raw_pdf_elements)[0]
    tables = data_category(raw_pdf_elements)[1]

    ##DEBUG
    # texts = texts[:1]
    # tables = tables[:1]

    table_summaries = tables_summarize(tables)
    text_summaries = texts

    if config.process_image:
        # Store base64 encoded images
        img_base64_list = []
        # Store image summaries
        image_summaries = []

        # Prompt
        img_prompt = "Describe the image in detail. Be specific about graphs, such as bar plots."
        path_figures = path + 'figures'

        # Read images, encode to base64 strings
        for img_file in sorted(os.listdir(path_figures)):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(path_figures, img_file)
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                img_cap = image_captioning(base64_image,img_prompt)
                time.sleep(60)
                image_summaries.append(img_cap)

        ##DEBUG
        # img_base64_list = img_base64_list[:1]
        # image_summaries = image_summaries[:1]



    # Add raw docs and doc summaries to Multi Vector Retriever.
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="multi_modal_rag",
                        embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    if config.process_image:
        # Add image summaries
        img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, img_base64_list)))



    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

    # RAG pipeline
    chain = (
        {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
        | RunnableLambda(prompt_func)
        | model
        | StrOutputParser()
    )

    response = chain.invoke(
    "Who is resposible for wildfires?"
    )
    print(response)


if __name__ == "__main__":
    main()


###### code to extract sources
# prompt = hub.pull("rlm/rag-prompt")

# def format_docs(docs):
#     return "\n\n".join(doc for doc in docs)

# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
#     | prompt
#     | model
#     | StrOutputParser()
# )

# rag_chain_with_source = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)



# rag_chain_with_source.invoke("What is the change in wild fires from 1993 to 2022 include chart?")