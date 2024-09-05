from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.embeddings.openai import OpenAIEmbeddings 
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma


load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"], temperature=0.5)

def AiResponse(question):
    prompt = PromptTemplate(
        input_variables=["question", "documents", "language"],
        template="""You are Omnidoc AI assistant, a specialized AI assistant in healthcare.

        Instructions:
        1. Only respond to questions related to healthcare or Omnidoc.
        2. For general health questions, respond directly without using the provided documents.
        3. For questions specific to Omnidoc, rely on the provided documents.
        4. Always recommend consulting a healthcare professional for personalized medical advice.
        5. Do not provide diagnoses or treatment plans.
        6. Avoid discussing or giving opinions on topics unrelated to healthcare.
        7. Respond concisely and directly.
        8. If you cannot assist, simply say: "I cannot answer this question."

        Question: {question}

        Relevant Documents:
        {documents}

        Respond in {language}.

        Answer:
        """
    )

    prompt_chain = prompt | llm | StrOutputParser()

    analyse_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an assistant designed to help users retrieve the most relevant documents from a vector database.
        Your task is to generate five distinct variations of the user's query to improve search results.
        These variations should capture different perspectives and nuances of the original question to mitigate the limitations of distance-based similarity searches.
        You provide all these variations under the "variations" field and the language of the question under the "language" field (do not use abbreviations).
        The question should be in JSON format.
        Do not add anything else.

        Original Question: {question}
        """
    )

    persist_directory = "vectorstore/"
    
    # OpenAIEmbeddings
    embd = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embd)
    
    analysis_chain = analyse_prompt | llm | JsonOutputParser()
    
    analysis_result = analysis_chain.invoke({"question": question})
    
    language = analysis_result["language"]
        
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=analyse_prompt
    )
    
    documents = retriever.invoke(question)

    response = prompt_chain.invoke({"question": question, "documents": documents, "language": language})
    
    return response

# Example usage
if __name__ == "__main__":
    response = AiResponse("When was Omnidoc founded?")
    print(response)

