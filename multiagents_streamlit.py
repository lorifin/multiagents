import streamlit as st
import os
from datetime import datetime
import json

# Import your multiagent service
from multiagents_haystack import ProductionMultiAgentService, ProductionConfig

# Configuration Streamlit
st.set_page_config(
    page_title="Assistant Multi-Agent",
    page_icon="🤖",
    layout="wide"
)

# CSS pour styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}
.agent-badge {
    background-color: #f0f2f6;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.8rem;
    font-weight: bold;
}
.response-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'service' not in st.session_state:
    st.session_state.service = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_service():
    """Initialize the multi-agent service"""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if not api_key or api_key == "your-openai-api-key-here":
            st.error("⚠️ Clé OpenAI non configurée. Ajoutez-la dans les secrets Streamlit.")
            return None
        
        config = ProductionConfig(
            openai_api_key=api_key,
            log_level="INFO"
        )
        
        with st.spinner("🚀 Initialisation du service multi-agent..."):
            service = ProductionMultiAgentService(config)
        
        st.success("✅ Service initialisé avec succès!")
        return service
        
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Assistant Multi-Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize service button
        if st.button("🚀 Initialiser le service"):
            st.session_state.service = initialize_service()
        
        # Status
        if st.session_state.service:
            st.success("🟢 Service actif")
            
            # Health check
            if st.button("🏥 Vérification santé"):
                health = st.session_state.service.health_check()
                st.json(health)
            
            # Metrics
            if st.button("📊 Métriques"):
                metrics = st.session_state.service.get_metrics()
                st.json(metrics)
        else:
            st.warning("🟡 Service non initialisé")
        
        # Clear chat
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main interface
    if not st.session_state.service:
        st.info("👆 Cliquez sur 'Initialiser le service' dans la barre latérale pour commencer")
        return
    
    # Chat interface
    st.header("💬 Chat avec l'assistant")
    
    # Display chat history
    for entry in st.session_state.chat_history:
        # User message
        st.markdown(f"**👤 Vous:** {entry['query']}")
        
        # Agent response
        agent_color = {
            "FAQ": "#17a2b8",
            "Retours": "#28a745", 
            "Produits": "#ffc107"
        }.get(entry['agent'], "#6c757d")
        
        st.markdown(f"""
        <div class="response-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span class="agent-badge" style="background-color: {agent_color}; color: white;">
                    🤖 {entry['agent']}
                </span>
                <small style="color: #6c757d;">
                    ⏱️ {entry['execution_time']:.2f}s
                </small>
            </div>
            <div style="margin-top: 0.5rem;">
                {entry['response']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Input form
    with st.form("query_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "Posez votre question:",
                placeholder="Ex: Comment faire un retour produit ?"
            )
        
        with col2:
            submitted = st.form_submit_button("Envoyer", use_container_width=True)
    
    # Process query
    if submitted and user_query:
        with st.spinner("🤔 L'assistant réfléchit..."):
            result = st.session_state.service.route_query(user_query)
        
        # Add to history
        entry = {
            "query": user_query,
            "agent": result['agent'],
            "response": result['response'],
            "execution_time": result['execution_time'],
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_history.append(entry)
        
        # Rerun to display new message
        st.rerun()
    
    # Quick examples
    st.header("💡 Exemples de questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❓ Livraison gratuite ?", use_container_width=True):
            if st.session_state.service:
                result = st.session_state.service.route_query("Est-ce que la livraison est gratuite ?")
                entry = {
                    "query": "Est-ce que la livraison est gratuite ?",
                    "agent": result['agent'],
                    "response": result['response'],
                    "execution_time": result['execution_time'],
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(entry)
                st.rerun()
    
    with col2:
        if st.button("🔄 Comment faire un retour ?", use_container_width=True):
            if st.session_state.service:
                result = st.session_state.service.route_query("Comment faire un retour produit ?")
                entry = {
                    "query": "Comment faire un retour produit ?",
                    "agent": result['agent'],
                    "response": result['response'],
                    "execution_time": result['execution_time'],
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(entry)
                st.rerun()
    
    with col3:
        if st.button("🛍️ Produits pour l'été ?", use_container_width=True):
            if st.session_state.service:
                result = st.session_state.service.route_query("Quels produits recommandez-vous pour l'été ?")
                entry = {
                    "query": "Quels produits recommandez-vous pour l'été ?",
                    "agent": result['agent'],
                    "response": result['response'],
                    "execution_time": result['execution_time'],
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(entry)
                st.rerun()

if __name__ == "__main__":
    main()