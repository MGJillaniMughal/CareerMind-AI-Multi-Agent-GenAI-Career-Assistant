
import os
import inspect
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_pills import pills
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import shutil
import uuid
from datetime import datetime
from typing import Callable, TypeVar
import plotly.express as px
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
from utils import get_system_info, validate_api_keys

load_dotenv()

# Configure page
st.set_page_config(
    page_title="CareerMind AI - Multi-Agent Career Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/careermind-ai',
        'Report a bug': "https://github.com/your-repo/careermind-ai/issues",
        'About': "CareerMind AI - Your intelligent career companion powered by multi-agent AI"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    
    .sidebar-section {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
    
    .debug-section {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set environment variables from secrets
def setup_environment():
    """Setup environment variables from Streamlit secrets"""
    env_vars = [
        "LINKEDIN_EMAIL", "LINKEDIN_PASS", "LANGCHAIN_API_KEY", 
        "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "GROQ_API_KEY",
        "SERPER_API_KEY", "FIRECRAWL_API_KEY", "LINKEDIN_SEARCH"
    ]
    
    for var in env_vars:
        os.environ[var] = st.secrets.get(var, os.getenv(var, ""))

setup_environment()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "active_option_index": None,
        "interaction_history": [],
        "response_history": ["🚀 Welcome to CareerMind AI! I'm your intelligent career assistant ready to help you navigate your professional journey."],
        "user_query_history": ["👋 Hello CareerMind AI!"],
        "session_id": str(uuid.uuid4()),
        "chat_started": datetime.now(),
        "total_interactions": 0,
        "user_preferences": {},
        "resume_uploaded": False,
        "api_configured": False,
        "debug_mode": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Header
st.markdown("""
<div style="
    background: linear-gradient(90deg, #667eea, #764ba2);
    padding: 2rem;
    border-radius: 15px;
    color: #ffffff;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🚀 <strong>CareerMind AI</strong></h1>
    <h3 style="margin: 0.5rem 0; font-weight: 500;">Multi-Agent GenAI Career Assistant</h3>
    <p style="opacity: 0.95; font-size: 1.1rem; max-width: 700px; margin: auto;">
        Your intelligent companion for career growth, powered by advanced AI agents
    </p>
    <p style="margin-top: 1.5rem; font-weight: 600; font-size: 1rem;">
        ⚡ Powered by 
        <a href="https://mgjillanimughal.github.io/" target="_blank" style="color: #00ffe0; text-decoration: none;">
            JillaniSofTech 😎
        </a>
    </p>
</div>
""", unsafe_allow_html=True)

# Analytics tracking
streamlit_analytics.start_tracking()

# Setup directories
temp_dir = "temp"
dummy_resume_path = os.path.abspath("Ali_Resume.pdf")

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    
    # File Upload Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Resume Upload")
    uploaded_document = st.file_uploader(
        "Upload Your Resume", 
        type=["pdf"],
        help="Upload your resume in PDF format for personalized analysis"
    )
    
    if uploaded_document:
        bytes_data = uploaded_document.read()
        filepath = os.path.join(temp_dir, "resume.pdf")
        with open(filepath, "wb") as f:
            f.write(bytes_data)
        st.session_state["resume_uploaded"] = True
        st.markdown('<div class="success-message">✅ Resume uploaded successfully!</div>', unsafe_allow_html=True)
    elif os.path.exists(dummy_resume_path):
        # Use dummy resume if available
        shutil.copy(dummy_resume_path, os.path.join(temp_dir, "resume.pdf"))
        st.session_state["resume_uploaded"] = True
        st.markdown('<div class="warning-message">📋 Using demo resume. Upload your own for personalized results.</div>', unsafe_allow_html=True)
    else:
        st.session_state["resume_uploaded"] = False
        st.markdown('<div class="warning-message">⚠️ Please upload your resume for full functionality.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Service Provider Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Model Configuration")
    
    service_provider = st.selectbox(
        "Select AI Provider",
        ("groq (llama-3.3-70b-versatile)", "openai", "anthropic (coming soon)"),
        help="Choose your preferred AI model provider"
    )
    
    # Model configuration based on provider
    if service_provider == "openai":
        api_key_openai = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("OPENAI_API_KEY", ""),
            type="password",
            help="Enter your OpenAI API key"
        )
        model_openai = st.selectbox(
            "OpenAI Model",
            ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo"),
        )
        settings = {
            "model": model_openai,
            "model_provider": "openai",
            "temperature": 0.3,
        }
        st.session_state["OPENAI_API_KEY"] = api_key_openai
        os.environ["OPENAI_API_KEY"] = api_key_openai
        st.session_state["api_configured"] = bool(api_key_openai)
        
    else:  # Groq
        if st.button("🔑 Configure Groq API Key (Optional)"):
            st.session_state["show_groq_input"] = True
            
        if st.session_state.get("show_groq_input", False):
            api_key_groq = st.text_input(
                "Groq API Key", 
                type="password",
                help="Enter your Groq API key for enhanced performance"
            )
            if api_key_groq:
                st.session_state["GROQ_API_KEY"] = api_key_groq
                os.environ["GROQ_API_KEY"] = api_key_groq
                st.session_state["api_configured"] = True
        
        settings = {
            "model": "llama-3.3-70b-versatile",
            "model_provider": "groq", 
            "temperature": 0.3,
        }
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        max_iterations = st.slider("Max Agent Iterations", 1, 10, 10)
        st.session_state["max_iterations"] = max_iterations
        enable_logging = st.checkbox("Enable Detailed Logging", value=False)
        st.session_state["debug_mode"] = st.checkbox("Enable Debug Mode", value=False)
        custom_instructions = st.text_area(
            "Custom Instructions", 
            placeholder="Add any specific preferences or requirements..."
        )
        
        if custom_instructions:
            st.session_state["user_preferences"]["custom_instructions"] = custom_instructions
    
    # System Information
    with st.expander("📊 System Info"):
        sys_info = get_system_info()
        st.json(sys_info)
    
    # Developer Tools Section
    st.markdown("---")
    st.markdown("### 🛠️ Developer Tools")
    
    if st.button("🧪 Test Agent Routing"):
        st.session_state["show_routing_test"] = True
    
    if st.button("📊 Show Session State"):
        st.session_state["show_session_state"] = True
    
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ["show_groq_input"]:  # Keep some keys
                del st.session_state[key]
        st.success("Session reset! Please refresh the page.")
        st.rerun()
    
    # GitHub and Support Links
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://github.com/MGJillaniMughal/CareerMind-AI-Multi-Agent-GenAI-Career-Assistant" target="_blank">
                ⭐ Star on GitHub
            </a><br>
            <a href="https://www.linkedin.com/in/jillanisofttech/" target="_blank">
                💼 Connect on LinkedIn
            </a><br>
            <a href="mailto:m.g.jillani123@gmail.com" target="_blank">
                📧 Support
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Content Area
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### 📈 Session Stats")
    
    # Session metrics
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.markdown(
            f'<div class="metric-card"><h4>{st.session_state["total_interactions"]}</h4><p>Interactions</p></div>',
            unsafe_allow_html=True
        )
    with col2_2:
        uptime = datetime.now() - st.session_state["chat_started"]
        st.markdown(
            f'<div class="metric-card"><h4>{uptime.seconds//60}m</h4><p>Session Time</p></div>',
            unsafe_allow_html=True
        )
    
    # Feature highlights
    st.markdown("### 🎯 Key Features")
    features = [
        "🔍 Smart Job Search",
        "📊 Resume Analysis", 
        "✍️ Cover Letter Generation",
        "🌐 Market Research",
        "📈 Career Path Planning",
        "💰 Salary Insights",
        "🎯 Skill Gap Analysis"
    ]
    
    for feature in features:
        st.markdown(f'<div class="feature-card">{feature}</div>', unsafe_allow_html=True)

# Initialize the multi-agent system
flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()

# Helper functions
def initialize_callback_handler(main_container: DeltaGenerator):
    """Initialize enhanced callback handler with context management"""
    V = TypeVar("V")

    def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
        context = get_script_run_ctx()

        def wrapped(*args, **kwargs) -> V:
            add_script_run_ctx(ctx=context)
            return func(*args, **kwargs)

        return wrapped

    streamlit_callback_instance = CustomStreamlitCallbackHandler(
        parent_container=main_container
    )

    for method_name, method in inspect.getmembers(
        streamlit_callback_instance, predicate=inspect.ismethod
    ):
        setattr(streamlit_callback_instance, method_name, wrap_function(method))

    return streamlit_callback_instance


def execute_enhanced_conversation(user_input: str, graph):
    """
    FIXED: Execute conversation with enhanced debugging and error handling
    """
    try:
        # Enhanced debugging
        logger.info(f"🟢 USER INPUT RECEIVED: '{user_input}'")
        
        callback_handler = initialize_callback_handler(st.container())
        
        # Create HumanMessage object properly
        from langchain_core.messages import HumanMessage
        user_message = HumanMessage(content=user_input)
        
        # Get max_iterations from advanced settings or use default
        max_iterations = st.session_state.get("max_iterations", 10)
        
        # Enhanced state with session information and better structure
        enhanced_state = {
            "messages": [user_message],  # Start with clean message list
            "user_input": user_input.strip(),    # Clean user input
            "config": settings,
            "callback": callback_handler,
            "session_id": st.session_state["session_id"],
            "user_preferences": st.session_state.get("user_preferences", {}),
            "error_count": 0,
            "resume_data": {},
            "job_preferences": {},
            "analysis_results": {}
        }
        
        # Debug information display
        logger.info(f"🟡 PROCESSING STATE: user_input='{enhanced_state['user_input']}', messages_count={len(enhanced_state['messages'])}")
        
        # Show processing status to user
        processing_container = st.container()
        with processing_container:
            st.info("🔄 **Processing your request with AI agents...**")
            
            # Add debug expander if debug mode is enabled
            if st.session_state.get("debug_mode", False):
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write(f"**User Input:** `{user_input}`")
                    st.write(f"**Input Length:** {len(user_input)} characters")
                    st.write(f"**State Keys:** {list(enhanced_state.keys())}")
                    st.write(f"**Config Model:** {settings.get('model', 'Unknown')}")
                    st.write("**Starting workflow execution...**")
        
        # Execute the graph with proper error handling
        try:
            output = graph.invoke(enhanced_state, {"recursion_limit": max_iterations})
            logger.info(f"🟣 WORKFLOW COMPLETED: output_type={type(output)}")
            
        except Exception as graph_error:
            logger.error(f"🔴 GRAPH EXECUTION ERROR: {str(graph_error)}")
            
            # Show error details in debug mode
            if st.session_state.get("debug_mode", False):
                with st.expander("🚨 Workflow Error Details", expanded=False):
                    st.error(f"Graph execution failed: {str(graph_error)}")
                    st.write(f"**Error Type:** {type(graph_error).__name__}")
                    st.write(f"**User Input:** {user_input}")
            
            return f"I encountered a technical issue while processing your request: {str(graph_error)}. Please try rephrasing your question or contact support if the issue persists."
        
        # Clear processing status
        processing_container.empty()
        
        # Enhanced output processing and validation
        if output and isinstance(output, dict) and "messages" in output:
            messages_list = output["messages"]
            logger.info(f"🟣 MESSAGES RECEIVED: {len(messages_list)} messages")
            
            # Find the last agent response (not from user)
            agent_response = None
            agent_name = "Unknown"
            
            for msg in reversed(messages_list):
                if hasattr(msg, 'name') and msg.name and msg.name not in ['Human', 'user']:
                    agent_response = msg
                    agent_name = msg.name
                    break
                elif hasattr(msg, 'content') and not hasattr(msg, 'name') and len(messages_list) > 1:
                    # This might be an AI response without explicit name
                    agent_response = msg
                    agent_name = "AI Assistant"
                    break
            
            if agent_response and hasattr(agent_response, 'content'):
                logger.info(f"🟢 RESPONSE FROM: {agent_name}")
                logger.info(f"🟢 RESPONSE LENGTH: {len(agent_response.content)} characters")
                
                # Update message history for continuity
                try:
                    message_history.clear()
                    message_history.add_messages(messages_list)
                except Exception as history_error:
                    logger.warning(f"Could not update message history: {str(history_error)}")
                
                # Update session statistics
                st.session_state["total_interactions"] += 1
                
                # Return the agent's response content
                response_content = agent_response.content
                
                # Validate response quality
                if len(response_content.strip()) < 20:
                    logger.warning("⚠️ Response too short, adding enhancement")
                    response_content += "\n\nIs there anything specific you'd like me to help you with next?"
                
                return response_content
                
            else:
                logger.error("🔴 NO VALID AGENT RESPONSE FOUND")
                # Debug the messages structure
                if st.session_state.get("debug_mode", False):
                    with st.expander("🔍 Message Debug Info", expanded=False):
                        for i, msg in enumerate(messages_list):
                            st.write(f"**Message {i}:** Type: {type(msg).__name__}")
                            if hasattr(msg, 'name'):
                                st.write(f"  - Name: {msg.name}")
                            if hasattr(msg, 'content'):
                                st.write(f"  - Content: {str(msg.content)[:100]}...")
                
                return "I processed your request but didn't receive a proper response from the AI agents. Please try rephrasing your question or contact support."
        
        else:
            logger.error("🔴 INVALID OUTPUT STRUCTURE")
            logger.error(f"Output type: {type(output)}")
            if isinstance(output, dict):
                logger.error(f"Output keys: {list(output.keys())}")
            
            return "I encountered an issue processing your request. The AI system didn't return a valid response. Please try again with a different question."
        
    except Exception as exc:
        logger.error(f"🔴 CRITICAL CONVERSATION ERROR: {str(exc)}")
        logger.error(f"Error type: {type(exc).__name__}")
        
        # Show comprehensive error information in debug mode
        if st.session_state.get("debug_mode", False):
            with st.expander("🚨 Critical Error Details", expanded=False):
                st.error(f"**Error Type:** {type(exc).__name__}")
                st.error(f"**Error Message:** {str(exc)}")
                st.write(f"**User Input:** `{user_input}`")
                st.write(f"**Input Length:** {len(user_input)}")
                st.write(f"**Session ID:** {st.session_state.get('session_id', 'Unknown')}")
                
                # Add troubleshooting tips
                st.write("**🔧 Troubleshooting Tips:**")
                st.write("- Try rephrasing your question")
                st.write("- Use simpler language")
                st.write("- Check if your resume is uploaded (for resume-related queries)")
                st.write("- Refresh the page and try again")
        
        return f"I encountered a critical error: {str(exc)}. Please try again with a simpler question, or refresh the page if the issue persists."


def test_supervisor_routing():
    """
    Test function to verify supervisor routing works correctly
    """
    st.write("## 🧪 Test Supervisor Routing")
    
    test_inputs = [
        "find software engineer jobs in San Francisco",
        "analyze my resume", 
        "write a cover letter for Google",
        "research Apple company",
        "career advice for data science",
        "salary trends for developers",
        "hello",
        "what can you do",
        "help me find jobs",
        "review my CV"
    ]
    
    st.write("**Testing routing decisions:**")
    
    try:
        # Import required modules
        from chains import get_supervisor_chain
        from llms import load_llm
        
        # Initialize the LLM and supervisor chain
        llm = load_llm(**settings)
        supervisor_chain = get_supervisor_chain(llm)
        
        results = []
        
        for test_input in test_inputs:
            try:
                from langchain_core.messages import HumanMessage
                result = supervisor_chain.invoke({"messages": [HumanMessage(content=test_input)]})
                
                # Expected routing for validation
                expected_routes = {
                    "find software engineer jobs in San Francisco": "JobSearcher",
                    "analyze my resume": "ResumeAnalyzer",
                    "write a cover letter for Google": "CoverLetterGenerator", 
                    "research Apple company": "WebResearcher",
                    "career advice for data science": "CareerAdvisor",
                    "salary trends for developers": "MarketAnalyst",
                    "hello": "ChatBot",
                    "what can you do": "ChatBot",
                    "help me find jobs": "JobSearcher",
                    "review my CV": "ResumeAnalyzer"
                }
                
                expected = expected_routes.get(test_input, "Unknown")
                actual = result.next_action
                is_correct = expected == actual
                
                status = "✅" if is_correct else "❌"
                
                results.append({
                    "input": test_input,
                    "expected": expected,
                    "actual": actual,
                    "correct": is_correct
                })
                
                st.write(f"{status} **Input:** `{test_input}`")
                st.write(f"   **Expected:** `{expected}` | **Actual:** `{actual}`")
                st.write("---")
                
            except Exception as e:
                st.write(f"❌ **Input:** `{test_input}` → **Error:** {str(e)}")
                st.write("---")
        
        # Summary
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        st.write(f"**📊 Routing Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)**")
        
        if accuracy < 80:
            st.error("⚠️ Routing accuracy is low. Check the supervisor chain configuration.")
        else:
            st.success("✅ Routing is working well!")
            
    except Exception as e:
        st.error(f"Error testing routing: {str(e)}")


def classify_user_request(user_input: str) -> str:
    """
    Classify the type of user request for better user feedback
    """
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['job', 'position', 'hiring', 'employment']):
        return "Job Search"
    elif any(word in user_lower for word in ['resume', 'cv', 'analyze']):
        return "Resume Analysis"
    elif any(word in user_lower for word in ['cover letter', 'application letter']):
        return "Cover Letter Generation"
    elif any(word in user_lower for word in ['research', 'company', 'information']):
        return "Research & Intelligence"
    elif any(word in user_lower for word in ['career', 'advice', 'guidance']):
        return "Career Guidance"
    elif any(word in user_lower for word in ['salary', 'market', 'trends']):
        return "Market Analysis"
    elif any(word in user_lower for word in ['hello', 'hi', 'help', 'what can you do']):
        return "General Assistance"
    else:
        return "General Query"


# Show routing test if requested
if st.session_state.get("show_routing_test", False):
    test_supervisor_routing()
    if st.button("Close Routing Test"):
        st.session_state["show_routing_test"] = False
        st.rerun()

# Show session state if requested
if st.session_state.get("show_session_state", False):
    st.write("## 📊 Current Session State")
    st.json({
        "resume_uploaded": st.session_state.get("resume_uploaded", False),
        "api_configured": st.session_state.get("api_configured", False),
        "total_interactions": st.session_state.get("total_interactions", 0),
        "session_id": st.session_state.get("session_id", "Unknown")[:8] + "...",
        "debug_mode": st.session_state.get("debug_mode", False),
        "max_iterations": st.session_state.get("max_iterations", 10)
    })
    if st.button("Close Session State"):
        st.session_state["show_session_state"] = False
        st.rerun()

# Main conversation interface
with col1:
    st.markdown("### 💬 Conversation")
    
    # Quick action pills
    st.markdown("#### 🚀 Quick Actions")
    options = [
        "🔍 Analyze my resume comprehensively",
        "💼 Search for AI/ML jobs in USA", 
        "🌟 Find trending tech skills in 2025",
        "📝 Generate a tailored cover letter",
        "🏢 Research Google career opportunities",
        "📈 Analyze tech industry salary trends",
        "🎯 Create my career development plan",
        "🌐 Latest AI industry news and insights",
        "💡 Skill gap analysis for data science",
        "🚀 Startup job opportunities worldwide",
        "📊 Compare tech salaries across cities"
    ]

    icons = ["📊", "💼", "🌟", "📝", "🏢", "📈", "🎯", "🌐", "💡", "🚀", "💰"]

    selected_query = pills(
        "Choose a quick action or type your own question:",
        options,
        clearable=True,
        icons=icons,
        index=st.session_state["active_option_index"],
        key="action_pills",
    )
    
    if selected_query:
        st.session_state["active_option_index"] = options.index(selected_query)

    # Chat input form
    with st.form(key="enhanced_query_form", clear_on_submit=True):
        user_input_query = st.text_area(
            "Your Question:",
            value=(selected_query if selected_query else ""),
            placeholder="💭 Ask me anything about your career, job search, or professional development...",
            height=100,
            key="chat_input",
        )
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_query_button = st.form_submit_button(
                label="🚀 Send Message", 
                use_container_width=True,
                type="primary"
            )
        with col_clear:
            clear_chat_button = st.form_submit_button(
                label="🗑️ Clear Chat",
                use_container_width=True
            )

    # Handle form submissions
    if clear_chat_button:
        st.session_state["user_query_history"] = []
        st.session_state["response_history"] = []
        st.session_state["total_interactions"] = 0
        message_history.clear()
        st.rerun()

    if submit_query_button and user_input_query:
        # Enhanced validation checks with better messaging
        if not st.session_state["resume_uploaded"]:
            st.warning("📋 **Note:** Some features work better with an uploaded resume. You can still use job search, research, and general advice features.")
        
        if service_provider == "openai" and not st.session_state.get("OPENAI_API_KEY"):
            st.error("🔑 Please configure your OpenAI API key in the sidebar, or switch to Groq.")
        else:
            # Enhanced processing with better user feedback
            with st.spinner("🤖 CareerMind AI is analyzing your request..."):
                # Show what type of request this appears to be
                request_type = classify_user_request(user_input_query)
                st.info(f"🎯 **Detected Request Type:** {request_type}")
                
                # Process the query with enhanced error handling
                try:
                    chat_output = execute_enhanced_conversation(user_input_query, flow_graph)
                    
                    # Validate output quality
                    if chat_output and len(chat_output.strip()) > 20:
                        st.session_state["user_query_history"].append(user_input_query)
                        st.session_state["response_history"].append(chat_output)
                        st.session_state["active_option_index"] = None
                        
                        # Show success message
                        st.success("✅ **Response generated successfully!**")
                        st.rerun()
                    else:
                        st.error("❌ **Response too short or empty.** Please try rephrasing your question.")
                        
                except Exception as e:
                    st.error(f"❌ **Error processing request:** {str(e)}")
                    st.info("💡 **Try:** Using simpler language or check your API configuration.")

    # Display chat history with enhanced styling
    st.markdown("#### 💬 Chat History")
    
    if st.session_state["response_history"]:
        # Create scrollable chat container
        chat_container = st.container()
        
        with chat_container:
            for i in range(len(st.session_state["response_history"])):
                # User message
                if i < len(st.session_state["user_query_history"]):
                    message(
                        st.session_state["user_query_history"][i],
                        is_user=True,
                        key=f"user_{i}",
                        avatar_style="fun-emoji",
                    )
                
                # AI response
                message(
                    st.session_state["response_history"][i],
                    key=f"ai_{i}",
                    avatar_style="bottts",
                )
                
                # Add timestamp for recent messages
                if i == len(st.session_state["response_history"]) - 1:
                    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.info("👋 Start a conversation by selecting a quick action or typing your question above!")

# Footer with additional information
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("""
    **🎯 About CareerMind AI**  
    Advanced multi-agent AI system designed to accelerate your career growth through intelligent automation and personalized insights.
    """)

with col_footer2:
    st.markdown("""
    **🛠️ Powered By**  
    - LangChain & LangGraph
    - OpenAI & Groq Models  
    - Streamlit Framework
    - Multi-Agent Architecture
    """)

with col_footer3:
    st.markdown("""
    **📞 Support & Resources**  
    - [📖 Documentation](https://docs.careermind.ai)
    - [🐛 Report Issues](https://github.com/MGJillaniMughal/CareerMind-AI-Multi-Agent-GenAI-Career-Assistant)
    - [💬 Community](https://discord.gg/careermind)
    - [📧 Contact](mailto:m.g.jillani123@gmail.com)
    """)

# Stop analytics tracking
streamlit_analytics.stop_tracking()