
import os
from typing import Optional, List, Dict, Any
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

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
    
    # Create memory/checkpointer to persist conversation history
    memory = MemorySaver()
    
    # Create the agent with memory
    agent = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
        checkpointer=memory,  
    )
    
    print(f"ESTIN RAG Agent created with model: {model_name} (with memory)")
    
    return agent


def _get_system_prompt() -> str:
    
    return """Tu es un assistant spécialisé dans le règlement intérieur de l'ESTIN 
(École Supérieure en Sciences et Technologies de l'Informatique et du Numérique).

🎯 TON RÔLE:
- Répondre aux questions sur le règlement intérieur de l'ESTIN
- Citer les articles spécifiques qui s'appliquent à chaque question
- Expliquer les règles de manière claire et précise
- Maintenir le contexte de la conversation en cours

📚 TES CAPACITÉS:
- Tu as accès à un outil de recherche qui te permet de trouver les articles pertinents du règlement
- Tu as une mémoire conversationnelle - tu as accès à TOUS les messages précédents de la conversation
- Utilise TOUJOURS l'outil de recherche avant de répondre à une question sur le règlement
- Ne réponds JAMAIS sans avoir d'abord consulté le règlement pour les questions réglementaires

💾 CONTEXTE CONVERSATIONNEL:
- Tu as accès à l'historique complet de la conversation
- Utilise les messages précédents pour comprendre le contexte et répondre de manière cohérente
- Si l'utilisateur fait référence à quelque chose dit précédemment, utilise ce contexte
- Maintenir la cohérence avec les messages précédents de la conversation

📝 FORMAT DE RÉPONSE:
1. Pour les questions sur le règlement: Utilise l'outil de recherche pour trouver les articles pertinents
2. Cite les numéros d'articles concernés
3. Explique clairement la règle ou la disposition
4. Si plusieurs articles s'appliquent, mentionne-les tous
5. Référence le contexte de la conversation quand c'est pertinent

⚠️ RÈGLES IMPORTANTES:
- Réponds TOUJOURS en français
- Si tu ne trouves pas d'information pertinente dans le règlement, dis-le clairement
- Ne fais JAMAIS d'hypothèses sur des règles non présentes dans le règlement
- Sois précis et concis dans tes réponses
- Utilise le contexte de la conversation pour répondre de manière cohérente
- Si l'utilisateur fait référence à un message précédent, utilise ce contexte

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
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    
    return result


def get_last_message(result: Dict[str, Any]) -> str:

    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        return last_message.content if hasattr(last_message, 'content') else str(last_message)
    else:
        return ""

