"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Custom Callback Handler with Rich UI and Progress Tracking
"""

from typing import Any, Dict, List, Optional, Union
import streamlit as st
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema import BaseMessage
from langchain_core.callbacks.base import BaseCallbackHandler
import time
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedStreamlitCallbackHandler(BaseCallbackHandler):
    """
    Enhanced callback handler for CareerMind AI with improved UI elements,
    progress tracking, and detailed logging capabilities.
    """
    
    def __init__(self, parent_container, max_thought_containers: int = 4):
        """
        Initialize the enhanced callback handler.
        
        Args:
            parent_container: Streamlit container for displaying content
            max_thought_containers: Maximum number of thought containers to display
        """
        super().__init__()
        self._parent_container = parent_container
        self.max_thought_containers = max_thought_containers
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.current_agent = None
        self.step_count = 0
        self.total_tokens = 0
        self.agent_history = []
        
        # Enhanced UI elements
        self.progress_bar = None
        self.status_container = None
        self.metrics_container = None
        
        logger.info(f"Enhanced callback handler initialized with session: {self.session_id}")
    
    def write_agent_name(self, name: str) -> None:
        """
        Enhanced agent name display with better styling and status tracking.
        
        Args:
            name: Name of the current agent
        """
        try:
            self.current_agent = name
            self.step_count += 1
            self.agent_history.append({
                'agent': name,
                'timestamp': time.time(),
                'step': self.step_count
            })
            
            # Create enhanced agent display
            if hasattr(self, '_parent_container') and self._parent_container:
                agent_container = self._parent_container.container()
                
                with agent_container:
                    # Agent header with enhanced styling
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        padding: 1rem;
                        border-radius: 10px;
                        color: white;
                        margin: 1rem 0;
                        border-left: 5px solid #4CAF50;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="margin: 0; display: flex; align-items: center;">
                            🤖 <strong>{name}</strong> 
                            <span style="margin-left: auto; font-size: 0.8em; opacity: 0.8;">
                                Step {self.step_count}
                            </span>
                        </h4>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; opacity: 0.9;">
                            Processing your request with specialized AI capabilities...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress tracking
                    if self.step_count > 1:
                        progress_value = min(self.step_count / 5, 1.0)  # Assume max 5 steps
                        st.progress(progress_value)
                        
                    # Real-time metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Step", self.step_count)
                    with col2:
                        elapsed_time = time.time() - self.start_time
                        st.metric("Elapsed Time", f"{elapsed_time:.1f}s")
                    with col3:
                        st.metric("Active Agent", name.split()[-1])
            
            logger.info(f"Agent {name} activated at step {self.step_count}")
            
        except Exception as e:
            logger.error(f"Error writing agent name: {str(e)}")
            # Fallback to simple display
            if hasattr(self, '_parent_container') and self._parent_container:
                self._parent_container.write(f"🤖 **{name}**")
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """
        Enhanced LLM start callback with detailed tracking.
        
        Args:
            serialized: Serialized LLM information
            prompts: List of prompts being processed
            **kwargs: Additional arguments
        """
        try:
            if hasattr(self, '_parent_container') and self._parent_container:
                with self._parent_container:
                    with st.expander(f"🧠 LLM Processing - {self.current_agent or 'Unknown Agent'}", expanded=False):
                        st.write("**Model Information:**")
                        if 'name' in serialized:
                            st.write(f"- Model: `{serialized['name']}`")
                        if 'kwargs' in serialized:
                            model_kwargs = serialized['kwargs']
                            if 'model_name' in model_kwargs:
                                st.write(f"- Model Name: `{model_kwargs['model_name']}`")
                            if 'temperature' in model_kwargs:
                                st.write(f"- Temperature: `{model_kwargs['temperature']}`")
                        
                        st.write("**Processing Status:**")
                        st.write("⏳ Generating response...")
                        
                        # Show prompt count
                        st.write(f"📝 Processing {len(prompts)} prompt(s)")
            
            logger.info(f"LLM started for {self.current_agent} with {len(prompts)} prompts")
            
        except Exception as e:
            logger.error(f"Error in on_llm_start: {str(e)}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Enhanced LLM end callback with response analysis.
        
        Args:
            response: LLM response result
            **kwargs: Additional arguments
        """
        try:
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                if token_usage:
                    self.total_tokens += token_usage.get('total_tokens', 0)
                    
                    # Display token usage in a subtle way
                    if hasattr(self, '_parent_container') and self._parent_container:
                        with self._parent_container:
                            st.caption(f"💭 Tokens used: {token_usage.get('total_tokens', 0)} | "
                                     f"Total session: {self.total_tokens}")
            
            logger.info(f"LLM completed for {self.current_agent}")
            
        except Exception as e:
            logger.error(f"Error in on_llm_end: {str(e)}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """
        Enhanced tool start callback with detailed tool information.
        
        Args:
            serialized: Serialized tool information
            input_str: Tool input string
            **kwargs: Additional arguments
        """
        try:
            tool_name = serialized.get("name", "Unknown Tool")
            
            if hasattr(self, '_parent_container') and self._parent_container:
                with self._parent_container:
                    with st.expander(f"🔧 Tool Execution: {tool_name}", expanded=False):
                        st.write("**Tool Details:**")
                        st.write(f"- **Tool Name:** `{tool_name}`")
                        st.write(f"- **Description:** {serialized.get('description', 'No description available')}")
                        
                        st.write("**Input Parameters:**")
                        if input_str:
                            # Try to format the input nicely
                            try:
                                import json
                                formatted_input = json.loads(input_str)
                                st.json(formatted_input)
                            except:
                                st.code(input_str, language="text")
                        
                        st.write("⚙️ **Status:** Executing tool...")
                        
                        # Add a small progress indicator
                        progress_placeholder = st.empty()
                        with progress_placeholder:
                            st.info("🔄 Tool is running...")
            
            logger.info(f"Tool {tool_name} started for {self.current_agent}")
            
        except Exception as e:
            logger.error(f"Error in on_tool_start: {str(e)}")
    
    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Enhanced tool end callback with result display.
        
        Args:
            output: Tool output
            color: Display color (optional)
            observation_prefix: Observation prefix (optional)
            llm_prefix: LLM prefix (optional)
            **kwargs: Additional arguments
        """
        try:
            if hasattr(self, '_parent_container') and self._parent_container:
                with self._parent_container:
                    # Show completion status
                    st.success("✅ Tool execution completed")
                    
                    # Show output preview if not too long
                    if output and len(output) < 500:
                        with st.expander("📋 Tool Output Preview", expanded=False):
                            st.text(output[:200] + "..." if len(output) > 200 else output)
            
            logger.info(f"Tool completed for {self.current_agent}")
            
        except Exception as e:
            logger.error(f"Error in on_tool_end: {str(e)}")
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """
        Enhanced agent action callback with action tracking.
        
        Args:
            action: Agent action being performed
            **kwargs: Additional arguments
        """
        try:
            if hasattr(self, '_parent_container') and self._parent_container:
                with self._parent_container:
                    st.info(f"🎯 **Action:** {action.tool} | **Input:** {str(action.tool_input)[:100]}...")
            
            logger.info(f"Agent {self.current_agent} performing action: {action.tool}")
            
        except Exception as e:
            logger.error(f"Error in on_agent_action: {str(e)}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """
        Enhanced agent finish callback with completion summary.
        
        Args:
            finish: Agent finish information
            **kwargs: Additional arguments
        """
        try:
            if hasattr(self, '_parent_container') and self._parent_container:
                with self._parent_container:
                    st.success(f"🎉 **{self.current_agent}** completed successfully!")
                    
                    # Show summary metrics
                    total_time = time.time() - self.start_time
                    with st.expander("📊 Session Summary", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Steps", self.step_count)
                            st.metric("Session Time", f"{total_time:.1f}s")
                        with col2:
                            st.metric("Agents Used", len(set(h['agent'] for h in self.agent_history)))
                            st.metric("Total Tokens", self.total_tokens)
            
            logger.info(f"Agent {self.current_agent} finished successfully")
            
        except Exception as e:
            logger.error(f"Error in on_agent_finish: {str(e)}")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """
        Enhanced error handling with user-friendly error display.
        
        Args:
            error: Exception that occurred
            **kwargs: Additional arguments
        """
        try:
            if hasattr(self, '_parent_container') and self._parent_container:
                with self._parent_container:
                    st.error(f"❌ **Error in {self.current_agent or 'Unknown Agent'}:** {str(error)}")
                    
                    with st.expander("🔍 Error Details", expanded=False):
                        st.code(str(error), language="text")
                        st.write("**Troubleshooting Tips:**")
                        st.write("- Check your API keys and configuration")
                        st.write("- Ensure your resume is properly uploaded")
                        st.write("- Try rephrasing your question")
                        st.write("- Contact support if the issue persists")
            
            logger.error(f"Chain error in {self.current_agent}: {str(error)}")
            
        except Exception as e:
            logger.error(f"Error in on_chain_error: {str(e)}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current session.
        
        Returns:
            Dict containing session statistics and information
        """
        try:
            total_time = time.time() - self.start_time
            unique_agents = list(set(h['agent'] for h in self.agent_history))
            
            return {
                'session_id': self.session_id,
                'total_time': total_time,
                'total_steps': self.step_count,
                'total_tokens': self.total_tokens,
                'unique_agents': unique_agents,
                'agent_history': self.agent_history,
                'current_agent': self.current_agent,
                'start_time': self.start_time
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            return {}


class CustomStreamlitCallbackHandler(EnhancedStreamlitCallbackHandler):
    """
    Backwards compatibility wrapper for the enhanced callback handler.
    Maintains the original interface while providing enhanced functionality.
    """
    
    def __init__(self, parent_container, max_thought_containers: int = 4):
        """
        Initialize with backwards compatibility.
        
        Args:
            parent_container: Streamlit container for displaying content
            max_thought_containers: Maximum number of thought containers
        """
        super().__init__(parent_container, max_thought_containers)
        logger.info("CustomStreamlitCallbackHandler initialized with enhanced features")


# Utility functions for callback management
def create_enhanced_callback(container, enable_metrics: bool = True) -> CustomStreamlitCallbackHandler:
    """
    Factory function to create an enhanced callback handler.
    
    Args:
        container: Streamlit container for the callback
        enable_metrics: Whether to enable detailed metrics tracking
        
    Returns:
        CustomStreamlitCallbackHandler: Configured callback handler
    """
    try:
        callback = CustomStreamlitCallbackHandler(container)
        
        if enable_metrics:
            callback.enable_metrics = True
        
        logger.info("Enhanced callback handler created successfully")
        return callback
        
    except Exception as e:
        logger.error(f"Error creating enhanced callback: {str(e)}")
        # Fallback to basic handler
        return CustomStreamlitCallbackHandler(container)


def get_callback_metrics(callback: CustomStreamlitCallbackHandler) -> Dict[str, Any]:
    """
    Extract metrics from a callback handler.
    
    Args:
        callback: The callback handler to extract metrics from
        
    Returns:
        Dict containing callback metrics
    """
    try:
        if hasattr(callback, 'get_session_summary'):
            return callback.get_session_summary()
        else:
            # Basic metrics for older callback versions
            return {
                'session_id': getattr(callback, 'session_id', 'unknown'),
                'current_agent': getattr(callback, 'current_agent', None),
                'step_count': getattr(callback, 'step_count', 0)
            }
            
    except Exception as e:
        logger.error(f"Error getting callback metrics: {str(e)}")
        return {}


def format_callback_display(name: str, status: str = "active") -> str:
    """
    Format agent name display with enhanced styling.
    
    Args:
        name: Agent name to format
        status: Current status (active, completed, error)
        
    Returns:
        str: Formatted HTML string for display
    """
    try:
        status_colors = {
            'active': '#4CAF50',
            'completed': '#2196F3', 
            'error': '#F44336',
            'waiting': '#FF9800'
        }
        
        color = status_colors.get(status, '#4CAF50')
        
        return f"""
        <div style="
            background: linear-gradient(90deg, {color} 0%, {color}AA 100%);
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            color: white;
            margin: 0.5rem 0;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        ">
            🤖 {name} - {status.upper()}
        </div>
        """
        
    except Exception as e:
        logger.error(f"Error formatting callback display: {str(e)}")
        return f"🤖 {name}"