from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.routers import ConditionalRouter
from typing import List
import os

#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --- Configuration de base (remplace ta clé OpenAI !) ---

#OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

import streamlit as st

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not isinstance(OPENAI_API_KEY, str) or not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError(f"Clé OpenAI absente ou mal formatée : {OPENAI_API_KEY!r}")


# --- 1. Données simulées pour chaque domaine ---
faq_data = [
    "Vous pouvez suivre votre commande dans l'espace client.",
    "Livraison gratuite dès 50€ d'achat.",
    "Nous acceptons les paiements par carte bancaire et PayPal.",
    "Le service client est ouvert du lundi au vendredi de 9h à 18h."
]

return_data = [
    "Pour effectuer un retour, connectez-vous à votre compte puis cliquez sur 'Retourner un article'.",
    "Les remboursements sont effectués sous 7 jours ouvrés après réception du colis.",
    "Vous avez 30 jours pour retourner un article non conforme.",
    "Les frais de retour sont gratuits si l'article est défectueux."
]

product_data = [
    "Le casque audio modèle X est compatible Bluetooth et offre 20h d'autonomie.",
    "Pour l'été, nous recommandons le ventilateur portable SmartBreeze.",
    "Notre nouvelle gamme de smartphones propose des appareils photo haute résolution.",
    "Les ordinateurs portables de la gamme Pro sont parfaits pour le travail à distance."
]

class MultiAgentCustomerService:
    def __init__(self):
        self.faq_pipeline = self._build_rag_pipeline(faq_data, "FAQ")
        self.return_pipeline = self._build_rag_pipeline(return_data, "Retours")
        self.product_pipeline = self._build_rag_pipeline(product_data, "Produits")
        
    def _build_rag_pipeline(self, data: List[str], domain: str) -> Pipeline:
        """Construit un pipeline RAG pour un domaine spécifique"""
        
        # Créer les documents
        documents = [Document(content=text) for text in data]
        
        # Initialiser le document store et l'embedder
        document_store = InMemoryDocumentStore()
        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        
        # IMPORTANT: Initialiser le modèle d'embedding
        doc_embedder.warm_up()
        
        # Embedder les documents et les stocker
        embedded_docs = doc_embedder.run(documents)
        document_store.write_documents(embedded_docs["documents"])
        
        # Créer le pipeline RAG
        pipeline = Pipeline()
        
        # Components
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=2)
        
        # IMPORTANT: Initialiser le text embedder aussi
        text_embedder.warm_up()
        
        prompt_template = f"""
        Tu es un assistant spécialisé dans le domaine {domain}.
        Utilise les informations suivantes pour répondre à la question de l'utilisateur:
        
        Contexte:
        {{{{ documents }}}}
        
        Question: {{{{ query }}}}
        
        Réponse (en français, précise et utile):
        """
        
        prompt_builder = PromptBuilder(template=prompt_template)
       # generator = OpenAIGenerator(model="gpt-3.5-turbo")
        from haystack.components.generators import OpenAIGenerator
        print("API KEY VALUE:", OPENAI_API_KEY, type(OPENAI_API_KEY))

        generator = OpenAIGenerator(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

        
        # Connecter les composants
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("generator", generator)
        
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        
        return pipeline
    
    def route_query(self, query: str) -> str:
        """Route la requête vers l'agent approprié"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["retour", "remboursement", "rembourser", "échanger"]):
            return self._run_pipeline(self.return_pipeline, query, "Agent Retours")
        elif any(keyword in query_lower for keyword in ["produit", "recommander", "acheter", "casque", "ventilateur"]):
            return self._run_pipeline(self.product_pipeline, query, "Agent Produits")
        else:
            return self._run_pipeline(self.faq_pipeline, query, "Agent FAQ")
    
    def _run_pipeline(self, pipeline: Pipeline, query: str, agent_name: str) -> str:
        """Exécute un pipeline et retourne la réponse"""
        try:
            result = pipeline.run({
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query}
            })
            response = result["generator"]["replies"][0]
            return f"[{agent_name}] {response}"
        except Exception as e:
            return f"[{agent_name}] Désolé, je n'ai pas pu traiter votre demande. Erreur: {str(e)}"

# --- Fonction de routage simple (alternative sans pipeline complexe) ---
def simple_route_query(query: str) -> str:
    """Version simplifiée sans RAG - juste pour démonstration"""
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["retour", "remboursement"]):
        return "[Agent Retours] Pour effectuer un retour, connectez-vous à votre compte puis cliquez sur 'Retourner un article'. Les remboursements sont effectués sous 7 jours ouvrés."
    elif any(keyword in query_lower for keyword in ["produit", "recommander"]):
        return "[Agent Produits] Pour l'été, nous recommandons le ventilateur portable SmartBreeze. Le casque audio modèle X est également très populaire."
    else:
        return "[Agent FAQ] Vous pouvez suivre votre commande dans l'espace client. Livraison gratuite dès 50€ d'achat."

# --- Exemple d'utilisation ---
def main():
    print("=== Multi-Agent Customer Service Demo ===\n")
    
    # Version simplifiée (sans OpenAI API)
    print("Version simplifiée (sans API):")
    user_inputs = [
        "Comment faire un retour produit ?",
        "Quels produits recommandez-vous pour l'été ?",
        "Est-ce que la livraison est gratuite ?"
    ]
    
    for msg in user_inputs:
        print(f"\nQuestion: {msg}")
        response = simple_route_query(msg)
        print(f"Réponse: {response}")
    
    print("\n" + "="*50)
    print("Pour utiliser la version complète avec RAG:")
    print("1. Remplace 'your-openai-api-key-here' par ta vraie clé OpenAI")
    print("2. Installe les dépendances: pip install haystack-ai sentence-transformers")
    print("3. Décommente le code ci-dessous")
    
    # Version complète avec RAG (nécessite une clé OpenAI valide)
    
    print("\n\nVersion complète avec RAG:")
    service = MultiAgentCustomerService()
    
    for msg in user_inputs:
        print(f"\nQuestion: {msg}")
        response = service.route_query(msg)
        print(f"Réponse: {response}")
    

if __name__ == "__main__":
    main()