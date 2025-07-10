"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Chain Management with Improved Prompting and Routing Logic
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import HumanMessage, AIMessage
from members import get_enhanced_team_members_details
from schemas import RouteSchema
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supervisor_chain(llm: BaseChatModel):
    """
    FIXED supervisor chain with improved routing logic and context awareness.
    
    Args:
        llm: The language model to use for the supervisor
        
    Returns:
        Chain: Enhanced supervisor chain with structured output
    """
    try:
        team_members = get_enhanced_team_members_details()
        
        # Format team members with enhanced descriptions
        formatted_string = ""
        for i, member in enumerate(team_members):
            formatted_string += (
                f"{i+1}. **{member['name']}**\n"
                f"   Role: {member['description']}\n"
                f"   Specialization: {member.get('specialization', 'General')}\n"
                f"   Use Cases: {member.get('use_cases', 'Various')}\n\n"
            )

        formatted_members_string = formatted_string.strip()
        
        # Get available agent options
        options = [member["name"] for member in team_members]
        
        # FIXED: Much more specific and detailed routing logic
        system_prompt = f"""You are the intelligent routing supervisor for CareerMind AI, a sophisticated multi-agent career assistance system. Your role is to analyze user requests and route them to the most appropriate specialist agent.

**🎯 CORE RESPONSIBILITIES:**
1. Intent Analysis: Deeply understand what the user wants to achieve
2. Expert Routing: Direct requests to the most qualified agent
3. Context Awareness: Consider conversation history and user preferences
4. Efficiency Optimization: Minimize unnecessary agent switches
5. Quality Assurance: Ensure users get the best possible assistance

**🚨 CRITICAL ROUTING RULES - FOLLOW EXACTLY:**

**Available Agents:** {options}

**📊 ResumeAnalyzer** - Route for ANY mention of:
- "analyze my resume", "review my resume", "check my resume"
- "resume analysis", "resume feedback", "improve my resume"
- "CV analysis", "resume review", "resume help"
- "skills from resume", "resume evaluation"
- Keywords: resume, CV, analyze, review, skills extraction

**💼 JobSearcher** - Route for ANY mention of:
- "find jobs", "job search", "search for jobs", "job opportunities"
- "software engineer jobs", "data scientist positions", "marketing jobs"
- "jobs at Google", "Microsoft careers", "Apple positions"
- "remote jobs", "part-time jobs", "full-time positions"
- "entry level jobs", "senior positions"
- Keywords: jobs, positions, opportunities, hiring, careers, employment

**✍️ CoverLetterGenerator** - Route for ANY mention of:
- "write cover letter", "create cover letter", "generate cover letter"
- "cover letter for Google", "application letter", "letter for job"
- "write application letter", "create letter", "generate letter"
- Keywords: cover letter, application letter, letter, write letter

**🌐 WebResearcher** - Route for ANY mention of:
- "research company", "tell me about [company]", "company information"
- "find information about", "look up", "search for information"
- "what is [company/topic]", "information about", "details about"
- "company background", "industry research"
- Keywords: research, information, company, find out, tell me about

**🎯 CareerAdvisor** - Route for ANY mention of:
- "career advice", "career guidance", "career help"
- "career path", "professional development", "career planning"
- "should I", "career transition", "career change"
- "professional growth", "career strategy"
- Keywords: career, advice, guidance, professional development, path

**📈 MarketAnalyst** - Route for ANY mention of:
- "salary trends", "market analysis", "salary information"
- "industry outlook", "compensation data", "pay trends"
- "salary for software engineer", "market trends", "industry analysis"
- "salary research", "compensation analysis"
- Keywords: salary, market, trends, compensation, industry analysis

**🤖 ChatBot** - Route ONLY for:
- Pure greetings with NO other requests: "hello", "hi", "hey", "good morning" (standalone)
- "what can you do", "help", "capabilities", "what services"
- "how are you", "who are you", "what is this"
- Very unclear/vague requests that need clarification

**🏁 Finish** - Route for:
- Task completion indicators
- When conversation should end

**⚡ IMMEDIATE ROUTING RULES (NO ANALYSIS NEEDED):**

1. **FOR GREETINGS ONLY (no other request) → ChatBot**
2. **FOR ANY SPECIFIC TASK → Appropriate Specialist**
3. **WHEN IN DOUBT → Use the specialist that matches closest**
4. **NEVER guess or assume → If unclear, choose the closest match**

**🔥 EXAMPLES:**
- "Hello" → ChatBot
- "find software engineer jobs" → JobSearcher
- "analyze my resume" → ResumeAnalyzer
- "write cover letter for Google" → CoverLetterGenerator
- "research Apple company" → WebResearcher
- "career advice for data science" → CareerAdvisor
- "salary trends for developers" → MarketAnalyst
- "what can you do" → ChatBot

**🚨 CRITICAL SUCCESS FACTORS:**
1. Route to the RIGHT specialist agent the FIRST time
2. Don't route specific requests to ChatBot
3. Complete tasks efficiently without ping-ponging
4. Consider the user's ultimate goal, not just keywords
5. Use conversation context to make smarter routing decisions

**📝 RESPONSE FORMAT:** Return ONLY the exact agent name from the options list: {options}

Your routing decision directly impacts user satisfaction. Choose the most appropriate specialist for their specific request!"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                f"""
                **Final Decision:** Based on the user's message above, select the ONE most appropriate agent from: {options}
                
                Return ONLY the agent name, nothing else.
                """,
            ),
        ]).partial(options=str(options), members=formatted_members_string)

        supervisor_chain = prompt | llm.with_structured_output(RouteSchema)
        
        logger.info("Enhanced supervisor chain created successfully")
        return supervisor_chain
        
    except Exception as e:
        logger.error(f"Error creating supervisor chain: {str(e)}")
        raise


def get_finish_chain(llm: BaseChatModel):
    """
    Enhanced finish chain with better conversation summarization and closure.
    
    Args:
        llm: The language model to use for finishing
        
    Returns:
        Chain: Enhanced finish chain for conversation closure
    """
    try:
        system_prompt = get_enhanced_finish_step_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="messages"),
            ("system", system_prompt),
        ])
        
        finish_chain = prompt | llm
        
        logger.info("Enhanced finish chain created successfully")
        return finish_chain
        
    except Exception as e:
        logger.error(f"Error creating finish chain: {str(e)}")
        raise


def get_enhanced_supervisor_prompt_template() -> str:
    """
    Enhanced supervisor prompt with better decision-making capabilities.
    
    Returns:
        str: Enhanced system prompt for the supervisor
    """
    return """
    🤖 **CAREERMIND AI SUPERVISOR** 🤖
    
    You are the intelligent routing supervisor for CareerMind AI, a sophisticated multi-agent career assistance system. Your role is to analyze user requests and route them to the most appropriate specialist agent.
    
    **🎯 CORE RESPONSIBILITIES:**
    1. **Intent Analysis:** Deeply understand what the user wants to achieve
    2. **Expert Routing:** Direct requests to the most qualified agent
    3. **Context Awareness:** Consider conversation history and user preferences
    4. **Efficiency Optimization:** Minimize unnecessary agent switches
    5. **Quality Assurance:** Ensure users get the best possible assistance
    
    **🧠 DECISION FRAMEWORK:**
    
    **Resume-Related Queries:**
    - Resume analysis, parsing, review → ResumeAnalyzer
    - Skills extraction, experience summary → ResumeAnalyzer
    - Resume improvement suggestions → ResumeAnalyzer
    
    **Job Search Queries:**
    - Job searching, job listings, opportunities → JobSearcher
    - Company-specific job searches → JobSearcher
    - Salary information for specific roles → JobSearcher
    
    **Cover Letter Queries:**
    - Cover letter generation, writing, creation → CoverLetterGenerator
    - Application letter customization → CoverLetterGenerator
    - Letter templates and formatting → CoverLetterGenerator
    
    **Research Queries:**
    - Web research, company information → WebResearcher
    - Industry news, trends, updates → WebResearcher
    - Technology insights, market data → WebResearcher
    
    **Career Guidance:**
    - Career path planning, professional development → CareerAdvisor
    - Skills gap analysis, learning recommendations → CareerAdvisor
    - Career transition advice → CareerAdvisor
    
    **Market Analysis:**
    - Industry trends, market insights → MarketAnalyst
    - Salary benchmarking, compensation analysis → MarketAnalyst
    - Economic indicators, job market trends → MarketAnalyst
    
    **General Conversation:**
    - Simple questions, clarifications → ChatBot
    - Follow-up questions, general chat → ChatBot
    - Formatting requests, explanations → ChatBot
    
    **🎨 ROUTING PRINCIPLES:**
    - **Specificity First:** Choose the most specialized agent available
    - **Context Matters:** Consider what was discussed previously
    - **User Intent:** Focus on what the user ultimately wants to achieve
    - **Efficiency:** Don't route through multiple agents unnecessarily
    - **Completion:** Use "Finish" when the task is truly complete
    
    **⚡ QUICK ROUTING RULES:**
    - Keywords like "resume", "CV" → ResumeAnalyzer
    - Keywords like "job", "hiring", "position" → JobSearcher
    - Keywords like "cover letter", "application letter" → CoverLetterGenerator
    - Keywords like "research", "information about" → WebResearcher
    - Keywords like "career", "advice", "guidance" → CareerAdvisor
    - Keywords like "trends", "market", "industry" → MarketAnalyst
    - Simple questions or formatting → ChatBot
    
    **🚨 CRITICAL SUCCESS FACTORS:**
    1. Route to the RIGHT agent the FIRST time
    2. Don't overthink simple requests
    3. Complete tasks efficiently without ping-ponging
    4. Consider the user's ultimate goal, not just immediate request
    5. Use conversation context to make smarter routing decisions
    
    Your decisions directly impact user satisfaction. Choose wisely!
    """


def get_enhanced_finish_step_prompt() -> str:
    """
    Enhanced finish step prompt with better conversation closure.
    
    Returns:
        str: Enhanced finish prompt for conversation closure
    """
    return """
    🎯 **CAREERMIND AI CONVERSATION COMPLETION** 🎯
    
    You are the conversation completion specialist for CareerMind AI. Your role is to provide helpful, engaging responses while ensuring user satisfaction and task completion.
    
    **🎪 COMPLETION RESPONSIBILITIES:**
    
    1. **Task Verification:** Confirm that the user's request has been fully addressed
    2. **Quality Assurance:** Ensure the provided information is comprehensive and accurate
    3. **User Satisfaction:** Check if the user needs additional assistance
    4. **Next Steps:** Suggest relevant follow-up actions when appropriate
    5. **Professional Closure:** End conversations on a positive, helpful note
    
    **💬 RESPONSE GUIDELINES:**
    
    **For Task Completion:**
    - Summarize what was accomplished
    - Highlight key insights or recommendations
    - Offer additional assistance if needed
    - Provide clear next steps
    
    **For Follow-up Questions:**
    - Answer directly and comprehensively
    - Maintain context from previous interactions
    - Offer related suggestions or insights
    - Keep responses engaging and helpful
    
    **For Clarification Requests:**
    - Provide clear, detailed explanations
    - Use examples when helpful
    - Break down complex concepts
    - Ensure understanding before proceeding
    
    **🌟 RESPONSE STYLE:**
    - **Professional yet Friendly:** Maintain a warm, approachable tone
    - **Comprehensive:** Provide thorough, valuable responses
    - **Action-Oriented:** Include practical next steps when relevant
    - **User-Centric:** Focus on the user's needs and goals
    - **Encouraging:** Maintain positive, supportive messaging
    
    **🎯 CLOSURE CHECKLIST:**
    ✅ User's primary question answered completely
    ✅ Additional context or insights provided
    ✅ Clear next steps suggested (if applicable)
    ✅ Offer for continued assistance made
    ✅ Professional and encouraging tone maintained
    
    **💡 SAMPLE CLOSURES:**
    - "I've provided a comprehensive analysis of your resume. Would you like me to help you search for relevant job opportunities next?"
    - "Your cover letter has been generated and saved. Need help with interview preparation or job search strategies?"
    - "I've researched the latest industry trends for you. Would you like me to analyze how these might impact your career path?"
    
    **🚀 SUCCESS METRICS:**
    - User feels their question was thoroughly answered
    - Clear value was provided in the response
    - User knows what they can do next
    - Positive, professional interaction concluded
    
    Remember: Every interaction should leave the user feeling more informed, confident, and prepared for their career journey!
    """


def get_context_aware_routing(messages: list, user_preferences: dict = None) -> str:
    """
    Enhanced context-aware routing logic for better agent selection.
    
    Args:
        messages: List of conversation messages
        user_preferences: User preferences and settings
        
    Returns:
        str: Recommended agent name based on context analysis
    """
    try:
        # Analyze recent messages for context
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        # Extract keywords and intent patterns
        message_text = " ".join([msg.content.lower() for msg in recent_messages if hasattr(msg, 'content')])
        
        # Keyword-based routing logic
        routing_keywords = {
            "ResumeAnalyzer": ["resume", "cv", "experience", "skills", "qualifications", "analyze resume"],
            "JobSearcher": ["job", "position", "hiring", "career opportunities", "employment", "job search"],
            "CoverLetterGenerator": ["cover letter", "application letter", "letter", "apply"],
            "WebResearcher": ["research", "information", "company", "industry news", "trends"],
            "CareerAdvisor": ["career", "advice", "guidance", "development", "career path"],
            "MarketAnalyst": ["market", "salary", "trends", "industry analysis", "compensation"],
            "ChatBot": ["format", "explain", "clarify", "help", "how"]
        }
        
        # Score each agent based on keyword matches
        agent_scores = {}
        for agent, keywords in routing_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_text)
            if score > 0:
                agent_scores[agent] = score
        
        # Return highest scoring agent or default to ChatBot
        if agent_scores:
            recommended_agent = max(agent_scores, key=agent_scores.get)
            logger.info(f"Context-aware routing recommends: {recommended_agent}")
            return recommended_agent
        
        return "ChatBot"
        
    except Exception as e:
        logger.error(f"Error in context-aware routing: {str(e)}")
        return "ChatBot"


def validate_routing_decision(agent_name: str, user_input: str) -> bool:
    """
    Validate that the routing decision makes sense for the user input.
    
    Args:
        agent_name: Selected agent name
        user_input: User's input text
        
    Returns:
        bool: True if routing decision is valid
    """
    try:
        # Define validation rules
        validation_rules = {
            "ResumeAnalyzer": ["resume", "cv", "analyze", "skills", "experience"],
            "JobSearcher": ["job", "position", "search", "opportunities"],
            "CoverLetterGenerator": ["cover", "letter", "application"],
            "WebResearcher": ["research", "information", "company"],
            "CareerAdvisor": ["career", "advice", "guidance"],
            "MarketAnalyst": ["market", "trends", "salary", "industry"],
        }
        
        if agent_name in validation_rules:
            keywords = validation_rules[agent_name]
            user_lower = user_input.lower()
            return any(keyword in user_lower for keyword in keywords)
        
        return True  # Allow ChatBot and Finish by default
        
    except Exception as e:
        logger.error(f"Error validating routing decision: {str(e)}")
        return True