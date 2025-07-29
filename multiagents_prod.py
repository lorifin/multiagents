import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from pathlib import Path

from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.routers import ConditionalRouter

#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import streamlit as st
import os

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Configuration pour la production
@dataclass
class ProductionConfig:
    openai_api_key: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: int = 30
    data_path: str = "data/"
    logs_path: str = "logs/"

class ProductionLogger:
    """Logger centralisé pour la production"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        os.makedirs(config.logs_path, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{config.logs_path}/multiagent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_query(self, query: str, agent: str, response: str, execution_time: float):
        """Log les interactions pour analytics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "agent": agent,
            "response": response[:100] + "..." if len(response) > 100 else response,
            "execution_time": execution_time
        }
        self.logger.info(f"INTERACTION: {json.dumps(log_entry)}")

class DataManager:
    """Gestionnaire de données pour la production"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.data_path = Path(config.data_path)
        self.data_path.mkdir(exist_ok=True)
    
    def load_domain_data(self, domain: str) -> List[str]:
        """Charge les données d'un domaine depuis un fichier"""
        file_path = self.data_path / f"{domain.lower()}_data.json"
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Erreur chargement données {domain}: {e}")
                return self._get_default_data(domain)
        else:
            # Créer fichier par défaut
            default_data = self._get_default_data(domain)
            self.save_domain_data(domain, default_data)
            return default_data
    
    def save_domain_data(self, domain: str, data: List[str]):
        """Sauvegarde les données d'un domaine"""
        file_path = self.data_path / f"{domain.lower()}_data.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_default_data(self, domain: str) -> List[str]:
        """Données par défaut pour chaque domaine"""
        defaults = {
            "FAQ": [
                "Vous pouvez suivre votre commande dans l'espace client.",
                "Livraison gratuite dès 50€ d'achat.",
                "Nous acceptons les paiements par carte bancaire et PayPal.",
                "Le service client est ouvert du lundi au vendredi de 9h à 18h."
            ],
            "Retours": [
                "Pour effectuer un retour, connectez-vous à votre compte puis cliquez sur 'Retourner un article'.",
                "Les remboursements sont effectués sous 7 jours ouvrés après réception du colis.",
                "Vous avez 30 jours pour retourner un article non conforme.",
                "Les frais de retour sont gratuits si l'article est défectueux."
            ],
            "Produits": [
                "Le casque audio modèle X est compatible Bluetooth et offre 20h d'autonomie.",
                "Pour l'été, nous recommandons le ventilateur portable SmartBreeze.",
                "Notre nouvelle gamme de smartphones propose des appareils photo haute résolution.",
                "Les ordinateurs portables de la gamme Pro sont parfaits pour le travail à distance."
            ]
        }
        return defaults.get(domain, [])

class ProductionMultiAgentService:
    """Version production du service multi-agent"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = ProductionLogger(config)
        self.data_manager = DataManager(config)
        
        # Initialisation des embedders (une seule fois)
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=config.embedding_model
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=config.embedding_model
        )
        
        # Warm up des modèles
        self._warm_up_models()
        
        # Création des pipelines
        self.pipelines = self._build_all_pipelines()
        
        # Métriques
        self.metrics = {
            "total_queries": 0,
            "queries_by_agent": {"FAQ": 0, "Retours": 0, "Produits": 0},
            "avg_response_time": 0.0
        }
    
    def _warm_up_models(self):
        """Initialise tous les modèles"""
        try:
            self.logger.logger.info("Initialisation des modèles d'embedding...")
            self.doc_embedder.warm_up()
            self.text_embedder.warm_up()
            self.logger.logger.info("Modèles initialisés avec succès")
        except Exception as e:
            self.logger.logger.error(f"Erreur initialisation modèles: {e}")
            raise
    
    def _build_all_pipelines(self) -> Dict[str, Pipeline]:
        """Construit tous les pipelines"""
        pipelines = {}
        domains = ["FAQ", "Retours", "Produits"]
        
        for domain in domains:
            try:
                data = self.data_manager.load_domain_data(domain)
                pipelines[domain] = self._build_rag_pipeline(data, domain)
                self.logger.logger.info(f"Pipeline {domain} créé avec {len(data)} documents")
            except Exception as e:
                self.logger.logger.error(f"Erreur création pipeline {domain}: {e}")
        
        return pipelines
    
    def _build_rag_pipeline(self, data: List[str], domain: str) -> Pipeline:
        """Construit un pipeline RAG optimisé pour la production"""
        
        # Créer les documents
        documents = [Document(content=text) for text in data]
        
        # Document store
        document_store = InMemoryDocumentStore()
        
        # Embedder les documents
        embedded_docs = self.doc_embedder.run(documents)
        document_store.write_documents(embedded_docs["documents"])
        
        # Pipeline
        pipeline = Pipeline()
        
        # Components
        retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=3)
        
        prompt_template = f"""
        Tu es un assistant expert du service client spécialisé en {domain}.
        
        Contexte disponible:
        {{{{ documents }}}}
        
        Question du client: {{{{ query }}}}
        
        Instructions:
        - Réponds de manière précise et professionnelle
        - Utilise uniquement les informations du contexte
        - Si tu ne peux pas répondre avec le contexte, dis-le clairement
        - Reste courtois et serviable
        
        Réponse:
        """
        
        prompt_builder = PromptBuilder(
            template=prompt_template, 
            required_variables=["documents", "query"]
        )
        from haystack.components.generators import OpenAIGenerator

        generator = OpenAIGenerator(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

                
        # Connexions
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("generator", generator)
        
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        
        return pipeline
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route et traite une requête avec gestion d'erreurs et métriques"""
        start_time = datetime.now()
        
        try:
            # Déterminer l'agent
            agent = self._determine_agent(query)
            
            # Exécuter le pipeline
            result = self._execute_pipeline(agent, query)
            
            # Calculer le temps d'exécution
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Mettre à jour les métriques
            self._update_metrics(agent, execution_time)
            
            # Logger l'interaction
            self.logger.log_query(query, agent, result["response"], execution_time)
            
            return {
                "success": True,
                "agent": agent,
                "response": result["response"],
                "execution_time": execution_time,
                "documents_used": len(result.get("documents", []))
            }
            
        except Exception as e:
            error_msg = f"Erreur traitement requête: {str(e)}"
            self.logger.logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "agent": "Error",
                "response": "Désolé, je ne peux pas traiter votre demande actuellement. Veuillez réessayer.",
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _determine_agent(self, query: str) -> str:
        """Détermine l'agent approprié avec logique améliorée"""
        query_lower = query.lower()
        
        # Mots-clés pour chaque domaine
        keywords = {
            "Retours": ["retour", "remboursement", "rembourser", "échanger", "défectueux", "non conforme"],
            "Produits": ["produit", "recommander", "acheter", "casque", "ventilateur", "smartphone", "ordinateur"],
            "FAQ": ["livraison", "paiement", "commande", "suivi", "horaire", "contact"]
        }
        
        # Score pour chaque agent
        scores = {}
        for agent, words in keywords.items():
            scores[agent] = sum(1 for word in words if word in query_lower)
        
        # Retourner l'agent avec le meilleur score, ou FAQ par défaut
        best_agent = max(scores, key=scores.get)
        return best_agent if scores[best_agent] > 0 else "FAQ"
    
    def _execute_pipeline(self, agent: str, query: str) -> Dict[str, Any]:
        """Exécute un pipeline avec retry et timeout"""
        pipeline = self.pipelines[agent]
        
        for attempt in range(self.config.max_retries):
            try:
                # Embedding de la requête
                query_embedding = self.text_embedder.run(query)
                
                # Exécution du pipeline
                result = pipeline.run({
                    "retriever": {"query_embedding": query_embedding["embedding"]},
                    "prompt_builder": {"query": query}
                })
                
                return {
                    "response": result["generator"]["replies"][0],
                    "documents": result["retriever"]["documents"]
                }
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                self.logger.logger.warning(f"Tentative {attempt + 1} échouée: {e}")
    
    def _update_metrics(self, agent: str, execution_time: float):
        """Met à jour les métriques"""
        self.metrics["total_queries"] += 1
        self.metrics["queries_by_agent"][agent] += 1
        
        # Moyenne mobile du temps de réponse
        total = self.metrics["total_queries"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = ((current_avg * (total - 1)) + execution_time) / total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques actuelles"""
        return self.metrics.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du système"""
        try:
            # Test simple
            test_result = self.route_query("Test de santé")
            
            return {
                "status": "healthy" if test_result["success"] else "unhealthy",
                "pipelines_loaded": len(self.pipelines),
                "total_queries_processed": self.metrics["total_queries"],
                "avg_response_time": self.metrics["avg_response_time"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# API REST avec FastAPI (optionnel)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Multi-Agent Customer Service API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    success: bool
    agent: str
    response: str
    execution_time: float

# Initialisation du service
config = ProductionConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
service = ProductionMultiAgentService(config)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    result = service.route_query(request.query)
    return QueryResponse(**result)

@app.get("/health")
async def health_check():
    return service.health_check()

@app.get("/metrics")
async def get_metrics():
    return service.get_metrics()


# Script principal pour tester
if __name__ == "__main__":
    # Configuration
    config = ProductionConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
        log_level="INFO"
    )
    
    try:
        # Initialisation du service
        print("🚀 Initialisation du service multi-agent...")
        service = ProductionMultiAgentService(config)
        print("✅ Service initialisé avec succès!")
        
        # Test de santé
        health = service.health_check()
        print(f"🏥 État de santé: {health}")
        
        # Tests
        test_queries = [
            "Comment faire un retour produit ?",
            "Quels produits recommandez-vous pour l'été ?",
            "Est-ce que la livraison est gratuite ?"
        ]
        
        print("\n🧪 Tests du système:")
        for query in test_queries:
            result = service.route_query(query)
            print(f"\n❓ Question: {query}")
            print(f"🤖 Agent: {result['agent']}")
            print(f"💬 Réponse: {result['response']}")
            print(f"⏱️  Temps: {result['execution_time']:.2f}s")
        
        # Métriques finales
        print(f"\n📊 Métriques: {service.get_metrics()}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")