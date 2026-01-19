
import os
from typing import Optional, List, Dict, Any
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from .tools import create_retrieval_tool


def create_estin_agent(
    groq_api_key: str,
    vector_store: PineconeVectorStore,
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.1,
    k: int = 2,
):
    # Set Groq API key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Initialize the LLM
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_retries=2,
    )
    
    # Create the retrieval tool
    retrieval_tool = create_retrieval_tool(vector_store, k=k)
    
    # Define the system prompt
    system_prompt = _get_system_prompt()
    
    # Create the agent
    agent = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
    )
    
    print(f"✅ ESTIN RAG Agent created with model: {model_name}")
    
    return agent


def _get_system_prompt() -> str:
    
    return """Tu es un assistant spécialisé dans le règlement intérieur de l'ESTIN 
(École Supérieure en Sciences et Technologies de l'Informatique et du Numérique).

🎯 TON RÔLE:
- Répondre aux questions sur le règlement intérieur de l'ESTIN
- Citer les articles spécifiques qui s'appliquent à chaque question
- Expliquer les règles de manière claire et précise

📚 TES CAPACITÉS:
- Tu as accès à un outil de recherche qui te permet de trouver les articles pertinents du règlement
- Utilise TOUJOURS l'outil de recherche avant de répondre à une question
- Ne réponds JAMAIS sans avoir d'abord consulté le règlement

📝 FORMAT DE RÉPONSE:
1. Utilise l'outil de recherche pour trouver les articles pertinents
2. Cite les numéros d'articles concernés
3. Explique clairement la règle ou la disposition
4. Si plusieurs articles s'appliquent, mentionne-les tous

⚠️ RÈGLES IMPORTANTES:
- Réponds TOUJOURS en français
- Si tu ne trouves pas d'information pertinente, dis-le clairement
- Ne fais JAMAIS d'hypothèses sur des règles non présentes dans le règlement
- Sois précis et concis dans tes réponses

🏫 CONTEXTE:
L'ESTIN est une école supérieure publique située à Béjaïa, Algérie.
Le règlement intérieur couvre:
- Les dispositions générales
- Les obligations du personnel enseignant
- Les obligations du personnel ATS et contractuel
- L'hygiène et la sécurité
- Le régime disciplinaire
- Les dispositions finales"""


def invoke_agent(
    agent,
    question: str,
    thread_id: str,
) -> Dict[str, Any]:

    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    
    return result


def get_last_message(result: Dict[str, Any]) -> str:
    """
    Extract the last message from an agent result.
    
    Args:
        result: The agent invocation result
        
    Returns:
        The content of the last message
    """
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        return last_message.content if hasattr(last_message, 'content') else str(last_message)
    return ""

