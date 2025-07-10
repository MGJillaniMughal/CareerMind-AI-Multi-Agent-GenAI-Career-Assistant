"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Agents Module with Improved Architecture and Features
"""

from typing import Any, TypedDict, List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import logging

from chains import get_finish_chain, get_supervisor_chain
from tools import (
    get_job_search_tool,
    ResumeExtractorTool,
    generate_letter_for_specific_job,
    get_google_search_results,
    save_cover_letter_for_specific_job,
    scrape_website,
    analyze_job_market_trends,
    career_path_analyzer,
    salary_analyzer_tool,
)
from prompts import (
    get_search_agent_prompt_template,
    get_analyzer_agent_prompt_template,
    researcher_agent_prompt_template,
    get_generator_agent_prompt_template,
    get_career_advisor_prompt_template,
    get_market_analyst_prompt_template,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class AgentState(TypedDict):
    """Enhanced state management for the multi-agent system"""
    user_input: str
    messages: List[BaseMessage]
    next_step: str
    config: dict
    callback: Any
    resume_data: dict
    job_preferences: dict
    analysis_results: dict
    error_count: int
    session_id: str


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str, agent_name: str = "Agent"):
    """
    Enhanced agent creation with better error handling and logging.
    
    Args:
        llm: LLM to be used to create the agent
        tools: List of tools to be given to the worker node
        system_prompt: System prompt to be used in the agent
        agent_name: Name of the agent for logging purposes
        
    Returns:
        AgentExecutor: The executor for the created agent
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent, 
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        logger.info(f"Successfully created {agent_name}")
        return executor
        
    except Exception as e:
        logger.error(f"Error creating {agent_name}: {str(e)}")
        raise


def supervisor_node(state: AgentState) -> AgentState:
    """
    FIXED supervisor node with enhanced routing logic and debugging.
    """
    try:
        chat_history = state.get("messages", [])
        user_input = state.get("user_input", "")
        
        # Enhanced debugging
        logger.info(f"🎯 SUPERVISOR PROCESSING INPUT: '{user_input}'")
        logger.info(f"📊 CURRENT MESSAGES COUNT: {len(chat_history)}")
        
        llm = init_chat_model(**state["config"])
        supervisor_chain = get_supervisor_chain(llm)
        
        # Ensure we have user input in messages format
        if not chat_history and user_input:
            chat_history = [HumanMessage(content=user_input)]
            logger.info("🔄 Created initial message from user input")
        elif user_input:
            # Check if user input is already in messages
            user_message_exists = any(
                hasattr(msg, 'content') and user_input in str(msg.content) 
                for msg in chat_history 
                if hasattr(msg, 'content')
            )
            if not user_message_exists:
                chat_history.append(HumanMessage(content=user_input))
                logger.info("➕ Added user input to chat history")
        
        # Get routing decision with enhanced error handling
        try:
            output = supervisor_chain.invoke({"messages": chat_history})
            routing_decision = output.next_action
            
            # Validate routing decision
            valid_agents = [
                "ResumeAnalyzer", "JobSearcher", "CoverLetterGenerator", 
                "WebResearcher", "CareerAdvisor", "MarketAnalyst", 
                "ChatBot", "Finish"
            ]
            
            if routing_decision not in valid_agents:
                logger.warning(f"⚠️ Invalid routing decision: {routing_decision}, defaulting to ChatBot")
                routing_decision = "ChatBot"
            
            logger.info(f"🚦 SUPERVISOR DECISION: '{user_input}' → {routing_decision}")
            
        except Exception as routing_error:
            logger.error(f"❌ Routing error: {str(routing_error)}")
            # Intelligent fallback based on keywords
            routing_decision = get_fallback_routing(user_input)
            logger.info(f"🔄 Fallback routing: {routing_decision}")
        
        # Update state
        state["next_step"] = routing_decision
        state["messages"] = chat_history
        state["error_count"] = 0
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Critical supervisor error: {str(e)}")
        state["next_step"] = "ChatBot"
        state["error_count"] = state.get("error_count", 0) + 1
        
        if state["error_count"] >= 3:
            state["next_step"] = "Finish"
        
        return state


def get_fallback_routing(user_input: str) -> str:
    """
    Intelligent fallback routing based on keyword analysis.
    
    Args:
        user_input: User's input text
        
    Returns:
        str: Agent name for routing
    """
    try:
        user_lower = user_input.lower()
        
        # Job search keywords
        if any(word in user_lower for word in ['job', 'position', 'hiring', 'career opportunities', 'employment']):
            return "JobSearcher"
        
        # Resume keywords
        elif any(word in user_lower for word in ['resume', 'cv', 'analyze', 'review', 'skills']):
            return "ResumeAnalyzer"
        
        # Cover letter keywords
        elif any(word in user_lower for word in ['cover letter', 'application letter', 'write letter']):
            return "CoverLetterGenerator"
        
        # Research keywords
        elif any(word in user_lower for word in ['research', 'company', 'information', 'find out']):
            return "WebResearcher"
        
        # Career advice keywords
        elif any(word in user_lower for word in ['career advice', 'guidance', 'career path']):
            return "CareerAdvisor"
        
        # Market analysis keywords
        elif any(word in user_lower for word in ['salary', 'market', 'trends', 'compensation']):
            return "MarketAnalyst"
        
        # Default to ChatBot for greetings and unclear requests
        else:
            return "ChatBot"
            
    except Exception as e:
        logger.error(f"Error in fallback routing: {str(e)}")
        return "ChatBot"


def job_search_node(state: AgentState) -> AgentState:
    """
    Enhanced job search node with comprehensive error handling and fallback mechanisms.
    """
    try:
        llm = init_chat_model(**state["config"])
        
        # Import the fixed job search tool
        from tools import get_job_search_tool
        search_agent = create_agent(
            llm, 
            [get_job_search_tool()], 
            get_search_agent_prompt_template(),
            "JobSearcher"
        )
        
        chat_history = state.get("messages", [])
        state["callback"].write_agent_name("🔍 JobSearcher Agent - Finding Opportunities")
        
        logger.info("🔍 JobSearcher agent starting job search...")
        
        try:
            # Execute the job search with timeout protection
            output = search_agent.invoke(
                {"messages": chat_history}, 
                {"callbacks": [state["callback"]]}
            )
            
            search_result = output.get("output", "")
            
            # Validate the output
            if not search_result or len(search_result.strip()) < 50:
                # Generate fallback response if output is too short
                search_result = generate_fallback_job_search_response(chat_history)
            
            # Check if the result indicates an error
            if "❌" in search_result or "error" in search_result.lower():
                logger.warning("Job search returned error, providing helpful guidance")
                search_result = enhance_error_response(search_result, chat_history)
            
            state["messages"].append(
                AIMessage(content=search_result, name="JobSearcher")
            )
            
            logger.info("✅ Job search completed successfully")
            return state
            
        except Exception as search_error:
            logger.error(f"❌ Job search execution error: {str(search_error)}")
            
            # Generate helpful fallback response
            fallback_message = generate_comprehensive_fallback_response(chat_history, str(search_error))
            
            state["messages"].append(
                AIMessage(content=fallback_message, name="JobSearcher")
            )
            
            return state
        
    except Exception as e:
        logger.error(f"❌ Critical error in job search node: {str(e)}")
        
        # Final fallback - always provide a helpful response
        emergency_response = f"""
🔍 **Job Search Assistant**

I'm here to help you find great job opportunities! While I encountered a technical issue, I can still assist you with:

🎯 **Job Search Strategies:**
- **Keyword Optimization:** Use specific job titles and skills in your search
- **Location Flexibility:** Consider remote work opportunities
- **Industry Focus:** Target specific industries that match your background
- **Company Research:** Research companies you're interested in

💼 **Popular Job Boards to Explore:**
- **LinkedIn Jobs:** Great for professional networking and opportunities
- **Indeed:** Comprehensive job listings across all industries
- **Glassdoor:** Company reviews and salary information
- **AngelList:** Startup opportunities and equity-based roles
- **Stack Overflow Jobs:** Tech-focused positions

🛠️ **Next Steps:**
1. **Update Your Profile:** Ensure your LinkedIn and resume are current
2. **Set Job Alerts:** Create alerts for positions matching your criteria
3. **Network Actively:** Reach out to connections in your target companies
4. **Prepare Applications:** Tailor your resume and cover letter for each application

💡 **How I Can Help:**
- Analyze your resume for specific roles
- Generate personalized cover letters
- Research companies and industries
- Provide career development advice

Would you like me to help with any of these areas? Please let me know what specific assistance you need!
"""
        
        state["messages"].append(
            AIMessage(content=emergency_response, name="JobSearcher")
        )
        
        return state


def generate_comprehensive_fallback_response(chat_history: list, error_msg: str) -> str:
    """
    Generate comprehensive fallback when all else fails.
    """
    return f"""
🎯 **CareerMind AI Job Search Assistant**

I'm here to help you with your job search, even though I encountered a technical issue ({error_msg[:50]}...).

🔍 **Let's approach this strategically:**

**1. Identify Your Goals:**
- What type of role are you seeking?
- Which industries interest you most?
- What's your preferred location or remote work preference?
- What salary range are you targeting?

**2. Optimize Your Search Strategy:**
- Use specific, relevant keywords
- Target companies that align with your values
- Leverage your professional network
- Apply to roles where you meet 70%+ of requirements

**3. Professional Resources:**
📱 **Job Search Apps:** LinkedIn, Indeed, Glassdoor
🌐 **Company Websites:** Apply directly for better visibility
🤝 **Networking Events:** Industry meetups and professional associations
📧 **Recruiters:** Connect with specialized recruiters in your field

**4. Application Excellence:**
- Customize your resume for each application
- Write compelling cover letters
- Follow up professionally
- Prepare thoroughly for interviews

💡 **How I Can Support You:**
- **Resume Review:** Analyze and optimize your resume
- **Cover Letter Creation:** Write persuasive application letters
- **Company Research:** Deep dive into potential employers
- **Interview Preparation:** Practice questions and strategies
- **Career Planning:** Long-term career development guidance

🚀 **Ready to Get Started?**
Tell me what specific aspect of your job search you'd like help with, and I'll provide detailed, actionable guidance!

Your career success is my priority. Let's work together to find your ideal opportunity!
"""


def generate_fallback_job_search_response(chat_history: list) -> str:
    """
    Generate a comprehensive fallback response when job search fails.
    """
    try:
        # Extract search intent from chat history
        user_query = ""
        for msg in reversed(chat_history):
            if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'name'):
                user_query = msg.content.lower()
                break
        
        # Determine job search keywords from query
        keywords = extract_job_keywords(user_query)
        location = extract_location_keywords(user_query)
        
        response = f"""
🔍 **Job Search Results & Guidance**

I understand you're looking for {keywords or 'job opportunities'}{"" if not location else f" in {location}"}. While I encountered some technical limitations with live job searching, I can provide you with comprehensive guidance and strategies.

🎯 **Recommended Search Strategy:**

**1. Primary Job Boards:**
- **LinkedIn Jobs:** Search for "{keywords or 'your target role'}" + your location
- **Indeed:** Use keywords like "{keywords or 'professional opportunities'}"
- **Glassdoor:** Great for company research and salary insights
- **Company Career Pages:** Apply directly to companies you admire

**2. Search Optimization Tips:**
✅ Use specific job titles (e.g., "Senior Software Engineer" vs "Developer")
✅ Include relevant skills and technologies in your search
✅ Set up job alerts for automatic notifications
✅ Consider remote work opportunities to expand your options

**3. Application Strategy:**
📝 **Resume Optimization:** Tailor your resume for each application
📝 **Cover Letter:** Write personalized cover letters highlighting relevant experience
📝 **Follow Up:** Apply within 48 hours of job posting for best visibility

💡 **Industry-Specific Guidance:**
{get_industry_specific_advice(keywords)}

🔗 **Networking Opportunities:**
- Professional associations in your field
- Industry meetups and conferences
- LinkedIn professional groups
- Alumni networks from your education

📊 **Market Insights:**
{get_market_insights(keywords)}

🎯 **Next Steps:**
1. **Immediate (Today):** Update your LinkedIn profile and resume
2. **This Week:** Apply to 5-10 relevant positions
3. **Ongoing:** Network with 2-3 new professionals weekly
4. **Monthly:** Review and adjust your search strategy

Would you like me to help you with:
- Resume analysis and optimization?
- Cover letter generation for specific roles?
- Company research and insights?
- Interview preparation strategies?

I'm here to support your job search journey!
"""
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating fallback response: {str(e)}")
        return get_basic_fallback_response()


def enhance_error_response(error_response: str, chat_history: list) -> str:
    """
    Enhance error responses with helpful guidance.
    """
    try:
        enhanced_response = f"""
{error_response}

🔧 **Let me help you with alternative approaches:**

**Immediate Solutions:**
1. **Broaden Your Search:** Try more general keywords
2. **Remove Location Filters:** Consider remote opportunities
3. **Check Spelling:** Ensure job titles and locations are correct
4. **Try Synonyms:** Use alternative terms for your target role

**Manual Search Recommendations:**
🎯 **Top Job Boards:**
- LinkedIn Jobs (linkedin.com/jobs)
- Indeed (indeed.com)
- Glassdoor (glassdoor.com)
- ZipRecruiter (ziprecruiter.com)

**Search Tips:**
✅ Use Boolean search operators (AND, OR, quotes)
✅ Set up job alerts for passive searching
✅ Search by company name for direct opportunities
✅ Use industry-specific job boards

**How I Can Still Help:**
- **Resume Analysis:** Upload your resume for optimization tips
- **Cover Letter Writing:** I can create personalized cover letters
- **Company Research:** Get insights about specific employers
- **Interview Prep:** Practice questions and answer strategies

Would you like assistance with any of these areas while we work on improving the job search functionality?
"""
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Error enhancing error response: {str(e)}")
        return error_response


def resume_analyzer_node(state: AgentState) -> AgentState:
    """
    Enhanced resume analyzer with detailed analysis capabilities.
    """
    try:
        llm = init_chat_model(**state["config"])
        analyzer_agent = create_agent(
            llm, 
            [ResumeExtractorTool(), career_path_analyzer()], 
            get_analyzer_agent_prompt_template(),
            "ResumeAnalyzer"
        )
        
        state["callback"].write_agent_name("📊 ResumeAnalyzer Agent - Analyzing Your Profile")
        
        logger.info("📊 ResumeAnalyzer agent starting analysis...")
        
        output = analyzer_agent.invoke(
            {"messages": state["messages"]}, 
            {"callbacks": [state["callback"]]}
        )
        
        # Store resume analysis results
        state["analysis_results"] = state.get("analysis_results", {})
        state["analysis_results"]["resume_analysis"] = output.get("output")
        
        state["messages"].append(
            AIMessage(content=output.get("output"), name="ResumeAnalyzer")
        )
        
        logger.info("✅ Resume analysis completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in resume analyzer node: {str(e)}")
        error_message = "I encountered an issue while analyzing your resume. Please ensure your resume is properly uploaded and try again."
        state["messages"].append(AIMessage(content=error_message, name="ResumeAnalyzer"))
        return state


def cover_letter_generator_node(state: AgentState) -> AgentState:
    """
    Enhanced cover letter generator with multiple templates and customization.
    """
    try:
        llm = init_chat_model(**state["config"])
        generator_agent = create_agent(
            llm,
            [
                generate_letter_for_specific_job,
                save_cover_letter_for_specific_job,
                ResumeExtractorTool(),
            ],
            get_generator_agent_prompt_template(),
            "CoverLetterGenerator"
        )

        state["callback"].write_agent_name("✍️ CoverLetterGenerator Agent - Crafting Your Letter")
        
        logger.info("✍️ CoverLetterGenerator agent starting...")
        
        output = generator_agent.invoke(
            {"messages": state["messages"]}, 
            {"callbacks": [state["callback"]]}
        )
        
        state["messages"].append(
            AIMessage(content=output.get("output"), name="CoverLetterGenerator")
        )
        
        logger.info("✅ Cover letter generation completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in cover letter generator node: {str(e)}")
        error_message = "I encountered an issue while generating your cover letter. Please ensure both resume and job details are available and try again."
        state["messages"].append(AIMessage(content=error_message, name="CoverLetterGenerator"))
        return state


def web_research_node(state: AgentState) -> AgentState:
    """
    Enhanced web research node with comprehensive research capabilities.
    """
    try:
        llm = init_chat_model(**state["config"])
        research_agent = create_agent(
            llm,
            [
                get_google_search_results, 
                scrape_website,
                analyze_job_market_trends,
            ],
            researcher_agent_prompt_template(),
            "WebResearcher"
        )
        
        state["callback"].write_agent_name("🌐 WebResearcher Agent - Gathering Intelligence")
        
        logger.info("🌐 WebResearcher agent starting research...")
        
        output = research_agent.invoke(
            {"messages": state["messages"]}, 
            {"callbacks": [state["callback"]]}
        )
        
        state["messages"].append(
            AIMessage(content=output.get("output"), name="WebResearcher")
        )
        
        logger.info("✅ Web research completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in web research node: {str(e)}")
        error_message = "I encountered an issue while conducting web research. Please try again with a more specific research query."
        state["messages"].append(AIMessage(content=error_message, name="WebResearcher"))
        return state


def career_advisor_node(state: AgentState) -> AgentState:
    """
    FIXED career advisor node for personalized career guidance.
    """
    try:
        llm = init_chat_model(**state["config"])
        
        # Import the fixed tools
        from tools import (
            ResumeExtractorTool,
            career_path_analyzer,
            analyze_job_market_trends,
            get_google_search_results,
        )
        
        # Create the tools list - notice no () after career_path_analyzer
        tools_list = [
            ResumeExtractorTool(),
            career_path_analyzer,  # This is now a proper @tool decorated function
            analyze_job_market_trends,  # This is now a proper @tool decorated function
            get_google_search_results,
        ]
        
        advisor_agent = create_agent(
            llm,
            tools_list,
            get_career_advisor_prompt_template(),
            "CareerAdvisor"
        )
        
        state["callback"].write_agent_name("🎯 CareerAdvisor Agent - Guiding Your Path")
        
        logger.info("🎯 CareerAdvisor agent starting guidance...")
        
        try:
            output = advisor_agent.invoke(
                {"messages": state["messages"]}, 
                {"callbacks": [state["callback"]]}
            )
            
            response_content = output.get("output", "")
            
            # If no output, provide helpful career guidance
            if not response_content or len(response_content.strip()) < 50:
                response_content = generate_fallback_career_advice(state.get("user_input", ""))
            
            state["messages"].append(
                AIMessage(content=response_content, name="CareerAdvisor")
            )
            
            logger.info("✅ Career advice generation completed successfully")
            return state
            
        except Exception as advisor_error:
            logger.error(f"❌ CareerAdvisor execution error: {str(advisor_error)}")
            
            # Generate helpful fallback advice
            fallback_advice = generate_comprehensive_career_advice(state.get("user_input", ""))
            
            state["messages"].append(
                AIMessage(content=fallback_advice, name="CareerAdvisor")
            )
            
            return state
        
    except Exception as e:
        logger.error(f"❌ Error in career advisor node: {str(e)}")
        
        # Provide comprehensive career guidance even on error
        emergency_advice = f"""
🎯 **Career Advisory Services**

I'm here to provide strategic career guidance! While I encountered a technical issue, I can still help you with:

## 🚀 **Career Development Strategies**

**📈 Career Path Planning:**
- Identify your strengths and interests
- Research target roles and industries
- Create a 3-5 year career roadmap
- Set achievable milestones and goals

**🎓 Skill Development:**
- Technical skills enhancement
- Leadership and soft skills
- Industry-specific certifications
- Continuous learning strategies

**🤝 Professional Networking:**
- LinkedIn optimization strategies
- Industry event participation
- Mentorship opportunities
- Professional relationship building

**💼 Career Transition Planning:**
- Industry change strategies
- Role advancement preparation
- Personal branding development
- Interview and negotiation skills

## 💡 **Immediate Next Steps:**

1. **Assess Your Current Position:**
   - Evaluate your skills and experience
   - Identify your career goals
   - Research your target industry

2. **Develop Your Action Plan:**
   - Create a learning roadmap
   - Set networking goals
   - Plan skill development activities

3. **Build Your Professional Brand:**
   - Update your LinkedIn profile
   - Create a portfolio of your work
   - Start sharing industry insights

## 🎯 **How I Can Help:**

- **Resume Analysis:** Review and optimize your resume
- **Job Search Strategy:** Find opportunities aligned with your goals
- **Cover Letter Creation:** Craft compelling application materials
- **Market Research:** Analyze industry trends and opportunities

Would you like specific guidance on any of these areas? Please let me know what aspect of your career development you'd like to focus on!
"""
        
        state["messages"].append(
            AIMessage(content=emergency_advice, name="CareerAdvisor")
        )
        
        return state

def generate_comprehensive_career_advice(user_input: str) -> str:
    """Generate comprehensive career advice as final fallback"""
    return f"""
🎯 **Comprehensive Career Development Guide**

Thank you for seeking career guidance! I'm here to help you navigate your professional journey successfully.

## 📈 **Career Development Framework**

**1. Self-Assessment & Goal Setting**
- Identify your values, interests, and strengths
- Define short-term and long-term career objectives
- Assess your current skills and experience level
- Determine your ideal work environment and culture

**2. Market Research & Opportunity Analysis**
- Research target industries and roles
- Understand current market trends and demands
- Identify growing sectors and emerging opportunities
- Analyze compensation trends and career trajectories

**3. Skill Development Strategy**
- Map required skills for your target roles
- Create a learning and development plan
- Pursue relevant certifications and training
- Build both technical and soft skills

**4. Professional Brand Building**
- Develop a compelling personal brand
- Optimize your online professional presence
- Create a portfolio showcasing your work
- Build thought leadership in your field

## 🛠️ **Practical Action Steps**

**Immediate Actions (This Week):**
- Update your LinkedIn profile with recent achievements
- Research 5 companies in your target industry
- Identify 3 skills to develop over the next 6 months
- Connect with 2 professionals in your desired field

**Short-term Goals (Next 3 Months):**
- Complete an online course or certification
- Attend 2 industry events or webinars
- Conduct 3 informational interviews
- Start a project that demonstrates your expertise

**Medium-term Objectives (6-12 Months):**
- Apply for roles that stretch your capabilities
- Publish content or speak at industry events
- Expand your professional network significantly
- Take on leadership responsibilities

## 💡 **Success Strategies**

**Networking Excellence:**
- Focus on giving value before receiving
- Maintain regular contact with your network
- Attend industry events and conferences
- Join professional associations in your field

**Continuous Learning:**
- Stay current with industry trends and technologies
- Seek feedback and mentorship opportunities
- Learn from both successes and failures
- Develop a growth mindset

**Personal Branding:**
- Be consistent across all professional platforms
- Share insights and expertise regularly
- Demonstrate your unique value proposition
- Build authentic relationships

## 🎯 **Specialized Guidance Available**

I can provide detailed assistance with:
- **Resume Analysis:** Optimize your resume for specific roles
- **Job Search Strategy:** Find and apply to relevant opportunities
- **Interview Preparation:** Practice and improve your interview skills
- **Salary Negotiation:** Research and negotiate competitive compensation
- **Career Transitions:** Plan and execute industry or role changes

What specific aspect of your career would you like to focus on next? I'm here to provide personalized guidance for your unique situation!
"""

def market_analyst_node(state: AgentState) -> AgentState:
    """
    Enhanced market analyst node for industry insights and trends.
    """
    try:
        llm = init_chat_model(**state["config"])
        analyst_agent = create_agent(
            llm,
            [
                get_google_search_results,
                scrape_website,
                analyze_job_market_trends,
                salary_analyzer_tool(),
            ],
            get_market_analyst_prompt_template(),
            "MarketAnalyst"
        )
        
        state["callback"].write_agent_name("📈 MarketAnalyst Agent - Analyzing Trends")
        
        logger.info("📈 MarketAnalyst agent starting analysis...")
        
        output = analyst_agent.invoke(
            {"messages": state["messages"]}, 
            {"callbacks": [state["callback"]]}
        )
        
        state["messages"].append(
            AIMessage(content=output.get("output"), name="MarketAnalyst")
        )
        
        logger.info("✅ Market analysis completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in market analyst node: {str(e)}")
        error_message = "I encountered an issue while analyzing market trends. Please try again with a specific market research question."
        state["messages"].append(AIMessage(content=error_message, name="MarketAnalyst"))
        return state

def generate_fallback_career_advice(user_input: str) -> str:
    """Generate fallback career advice when tools fail"""
    try:
        user_lower = user_input.lower() if user_input else ""
        
        # Detect career focus from input
        if "data science" in user_lower:
            focus = "data science"
        elif "software" in user_lower or "engineer" in user_lower:
            focus = "software engineering"
        elif "marketing" in user_lower:
            focus = "marketing"
        elif "management" in user_lower:
            focus = "management"
        else:
            focus = "professional development"
        
        advice = f"""
🎯 **Career Guidance: {focus.title()}**

Based on your interest in {focus}, here's strategic career guidance:

## 📊 **Current Market Assessment**

**Industry Outlook:**
The {focus} field shows strong growth potential with increasing demand for skilled professionals. Companies are actively seeking talent with both technical expertise and business acumen.

**Key Success Factors:**
✅ Continuous learning and skill development
✅ Building a strong professional network
✅ Demonstrating measurable impact in your work
✅ Staying current with industry trends and technologies

## 🛤️ **Career Path Recommendations**

**Short-term (6-12 months):**
- Identify and fill skill gaps through targeted learning
- Build a portfolio showcasing your expertise
- Expand your professional network within the industry
- Gain experience through projects or volunteer work

**Medium-term (1-3 years):**
- Pursue relevant certifications or advanced education
- Take on leadership responsibilities in current role
- Mentor others and build thought leadership
- Explore opportunities in growing market segments

**Long-term (3+ years):**
- Consider specialized expertise or management tracks
- Build industry recognition through speaking/writing
- Evaluate entrepreneurial or consulting opportunities
- Plan for executive or senior leadership roles

## 🎓 **Skill Development Priorities**

**Core Competencies:**
- Technical skills specific to {focus}
- Data analysis and decision-making
- Communication and presentation skills
- Project management and leadership

**Emerging Skills:**
- Digital transformation capabilities
- Cross-functional collaboration
- Agile/lean methodologies
- Customer-centric thinking

## 💡 **Actionable Next Steps**

1. **Skills Assessment:** Evaluate your current capabilities against market requirements
2. **Learning Plan:** Create a structured approach to skill development
3. **Network Building:** Connect with professionals in your target field
4. **Experience Gaining:** Seek projects that demonstrate your growing expertise

## 🚀 **Resources for Growth**

**Learning Platforms:**
- Online courses (Coursera, Udemy, LinkedIn Learning)
- Professional certifications
- Industry conferences and workshops
- Mentorship programs

**Networking Opportunities:**
- Industry professional associations
- Local meetups and events
- LinkedIn professional groups
- Alumni networks

Would you like me to dive deeper into any specific aspect of your career development?
"""
        
        return advice
        
    except Exception as e:
        logger.error(f"Error generating fallback career advice: {str(e)}")
        return "I'm here to help with your career development. Please let me know what specific guidance you're looking for!"



def chatbot_node(state: AgentState) -> AgentState:
    """
    FIXED chatbot node with better conversation handling and proper scope.
    """
    try:
        # Get the user's input from the state
        user_input = state.get("user_input", "")
        messages = state.get("messages", [])
        
        # Debug logging
        logger.info(f"🤖 CHATBOT PROCESSING: '{user_input}'")
        
        # If no user input, try to get from messages
        if not user_input and messages:
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'name'):
                    user_input = msg.content
                    break
        
        user_lower = user_input.lower().strip()
        
        # ONLY handle greetings and general help - NOT specific requests
        if any(greeting in user_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]) and len(user_input.split()) <= 3:
            response_content = """Hello! 👋 Welcome to CareerMind AI! 

I'm your intelligent career assistant, ready to help you with:

🔍 **Job Search** - Find relevant opportunities across multiple platforms
📊 **Resume Analysis** - Optimize your resume for ATS and specific roles  
✍️ **Cover Letters** - Create personalized, compelling application letters
🎯 **Career Advice** - Strategic guidance for your professional growth
📈 **Market Research** - Industry trends and salary insights
🌐 **Company Research** - Deep dive into potential employers

How can I assist you with your career goals today?"""

        elif "what can you do" in user_lower or "help" in user_lower or "capabilities" in user_lower:
            response_content = """I'm CareerMind AI, your comprehensive career assistant! Here's how I can help:

🎯 **My Specialized Capabilities:**

📊 **Resume Analyzer** - Upload your resume and I'll:
   • Analyze it for ATS compatibility
   • Extract and categorize your skills
   • Provide improvement recommendations
   • Assess market positioning

💼 **Job Searcher** - I can help you:
   • Find relevant job opportunities
   • Filter by location, experience, and type
   • Provide salary insights
   • Research companies

✍️ **Cover Letter Generator** - I'll create:
   • Personalized cover letters for specific jobs
   • ATS-optimized content
   • Professional templates
   • Multiple format outputs

🌐 **Research Specialist** - I can research:
   • Company backgrounds and culture
   • Industry trends and market data
   • Competitor analysis
   • Recent news and developments

🎯 **Career Advisor** - Get guidance on:
   • Career path planning
   • Skill development roadmaps
   • Professional growth strategies
   • Industry transition advice

📈 **Market Analyst** - Access insights on:
   • Salary benchmarking
   • Industry growth trends
   • Skills demand forecasting
   • Economic indicators

Just tell me what you'd like to work on, and I'll connect you with the right specialist!"""

        elif "who are you" in user_lower or "what is this" in user_lower:
            response_content = """I'm CareerMind AI, your intelligent multi-agent career assistant! 🚀

I'm powered by advanced AI technology and consist of specialized agents, each expert in different aspects of career development:

👥 **My Expert Team:**
- 🔍 **JobSearcher:** Finds the best job opportunities for you
- 📊 **ResumeAnalyzer:** Optimizes your resume for success
- ✍️ **CoverLetterGenerator:** Creates compelling application letters
- 🌐 **WebResearcher:** Gathers market intelligence
- 🎯 **CareerAdvisor:** Provides strategic career guidance
- 📈 **MarketAnalyst:** Analyzes industry trends and salaries

💡 **How I Work:**
When you ask me something, I automatically route your question to the most qualified specialist agent who can provide you with expert-level assistance.

🎯 **My Goal:**
To accelerate your career growth through intelligent automation and personalized insights.

What would you like help with today?"""

        else:
            # This should rarely happen if routing works correctly
            # But provide a helpful redirect
            response_content = f"""I understand you're asking about: "{user_input}"

I see this might be a specific request that would be better handled by one of my specialist agents. Let me help you get the right assistance:

🔍 **For job searching:** Try "find software engineer jobs in NYC"
📊 **For resume help:** Try "analyze my resume" 
✍️ **For cover letters:** Try "create cover letter for Google"
🌐 **For research:** Try "research Apple company"
🎯 **For career advice:** Try "career advice for data science"
📈 **For market data:** Try "salary trends for developers"

Please rephrase your request more specifically, and I'll connect you with the right expert!"""

        state["callback"].write_agent_name("🤖 ChatBot Agent - Here to Help")
        
        # Create the AI message response
        ai_response = AIMessage(content=response_content, name="ChatBot")
        state["messages"].append(ai_response)
        
        logger.info("✅ ChatBot response completed successfully")
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in chatbot node: {str(e)}")
        # Provide a fallback response
        error_message = "I'm here to help with your career needs! Please let me know what you'd like assistance with - job searching, resume analysis, cover letters, or career advice."
        ai_response = AIMessage(content=error_message, name="ChatBot")
        state["messages"].append(ai_response)
        return state


def define_graph() -> StateGraph:
    """
    FIXED graph definition with improved routing and termination logic.
    
    Returns:
        StateGraph: The compiled graph representing the enhanced workflow
    """
    try:
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("ResumeAnalyzer", resume_analyzer_node)
        workflow.add_node("JobSearcher", job_search_node)
        workflow.add_node("CoverLetterGenerator", cover_letter_generator_node)
        workflow.add_node("WebResearcher", web_research_node)
        workflow.add_node("CareerAdvisor", career_advisor_node)
        workflow.add_node("MarketAnalyst", market_analyst_node)
        workflow.add_node("ChatBot", chatbot_node)
        workflow.add_node("Supervisor", supervisor_node)

        # Define worker nodes (excluding Supervisor)
        members = [
            "ResumeAnalyzer",
            "CoverLetterGenerator", 
            "JobSearcher",
            "WebResearcher",
            "CareerAdvisor",
            "MarketAnalyst",
            "ChatBot",
        ]
        
        # Set entry point
        workflow.set_entry_point("Supervisor")

        # FIXED: Enhanced routing logic with better termination
        def should_continue(state):
            """Determine if we should continue or finish - ENHANCED VERSION"""
            next_step = state.get("next_step", "")
            
            logger.info(f"🔄 WORKFLOW DECISION: next_step = '{next_step}'")
            
            # Check if explicitly set to finish
            if next_step == "Finish":
                logger.info("🏁 Workflow finishing (explicit)")
                return "Finish"
            
            # Route to appropriate agents - they will handle and then terminate
            if next_step in members:
                logger.info(f"➡️ Routing to {next_step}")
                return next_step
            
            # Default to ChatBot for unknown or empty next_step
            logger.info("🤖 Defaulting to ChatBot")
            return "ChatBot"

        # Add conditional edges from Supervisor
        workflow.add_conditional_edges(
            "Supervisor",
            should_continue,
            {
                "ResumeAnalyzer": "ResumeAnalyzer",
                "CoverLetterGenerator": "CoverLetterGenerator", 
                "JobSearcher": "JobSearcher",
                "WebResearcher": "WebResearcher",
                "CareerAdvisor": "CareerAdvisor",
                "MarketAnalyst": "MarketAnalyst",
                "ChatBot": "ChatBot",
                "Finish": END
            }
        )

        # FIXED: All agents go directly to END after completing their task
        # This prevents infinite loops and ensures clean termination
        for member in members:
            workflow.add_edge(member, END)

        graph = workflow.compile()
        logger.info("✅ Successfully compiled FIXED multi-agent graph with improved termination")
        return graph
        
    except Exception as e:
        logger.error(f"❌ Error defining graph: {str(e)}")
        raise


# Helper functions for job search fallbacks
def extract_job_keywords(query: str) -> str:
    """Extract job-related keywords from user query."""
    try:
        job_terms = [
            'software engineer', 'data scientist', 'product manager', 'marketing manager',
            'developer', 'analyst', 'designer', 'consultant', 'manager', 'engineer',
            'specialist', 'coordinator', 'director', 'lead', 'senior', 'junior'
        ]
        
        query_lower = query.lower()
        found_terms = [term for term in job_terms if term in query_lower]
        
        if found_terms:
            return found_terms[0]
        
        # Extract potential job keywords
        words = query_lower.split()
        job_words = [word for word in words if len(word) > 3 and word not in ['jobs', 'find', 'search', 'looking', 'want']]
        
        return ' '.join(job_words[:2]) if job_words else 'professional opportunities'
        
    except:
        return 'job opportunities'


def extract_location_keywords(query: str) -> str:
    """Extract location-related keywords from user query."""
    try:
        locations = [
            'san francisco', 'new york', 'seattle', 'austin', 'boston', 'chicago',
            'los angeles', 'denver', 'remote', 'california', 'texas', 'washington'
        ]
        
        query_lower = query.lower()
        found_locations = [loc for loc in locations if loc in query_lower]
        
        return found_locations[0] if found_locations else ""
        
    except:
        return ""


def get_industry_specific_advice(keywords: str) -> str:
    """Get industry-specific advice based on keywords."""
    try:
        if not keywords:
            return "Focus on roles that match your skills and interests."
        
        keywords_lower = keywords.lower()
        
        if any(word in keywords_lower for word in ['software', 'developer', 'engineer', 'tech']):
            return """
**Technology Sector:**
- GitHub portfolio is essential for showcasing your code
- Stay current with latest frameworks and technologies
- Contribute to open source projects
- Consider both startups and established tech companies
- Remote work opportunities are abundant in tech"""
        
        elif any(word in keywords_lower for word in ['data', 'scientist', 'analyst']):
            return """
**Data & Analytics:**
- Build a portfolio of data projects on GitHub/Kaggle
- Showcase expertise in Python, R, SQL, and visualization tools
- Consider industries like finance, healthcare, and e-commerce
- Highlight experience with machine learning and statistics
- Demonstrate business impact of your analytical work"""
        
        elif any(word in keywords_lower for word in ['marketing', 'digital', 'social']):
            return """
**Marketing & Digital:**
- Build a personal brand on social media platforms
- Create a portfolio showcasing successful campaigns
- Stay updated with digital marketing trends and tools
- Consider both agency and in-house positions
- Highlight ROI and measurable results from your work"""
        
        else:
            return """
**General Professional Advice:**
- Tailor your application materials to each role
- Research company culture and values alignment
- Prepare specific examples demonstrating your impact
- Network within your industry and target companies
- Consider both traditional and emerging career paths"""
            
    except:
        return "Focus on highlighting your unique value proposition to employers."


def get_market_insights(keywords: str) -> str:
    """Get market insights based on keywords."""
    try:
        if not keywords:
            return "The job market shows strong demand across multiple sectors."
        
        keywords_lower = keywords.lower()
        
        if 'tech' in keywords_lower or 'software' in keywords_lower:
            return "Technology sector continues strong growth with high demand for skilled professionals."
        elif 'data' in keywords_lower:
            return "Data professionals are in high demand across all industries as companies become more data-driven."
        elif 'marketing' in keywords_lower:
            return "Digital marketing roles are expanding as businesses increase their online presence."
        else:
            return "Job market shows resilience with opportunities across various sectors and experience levels."
            
    except:
        return "Current job market offers diverse opportunities for qualified candidates."


def get_basic_fallback_response() -> str:
    """Basic fallback response for critical errors."""
    return """
🎯 **Job Search Assistance**

I'm here to help with your career goals! While experiencing technical issues, I can still provide:

✅ Resume analysis and optimization
✅ Cover letter writing assistance  
✅ Company research and insights
✅ Career planning and development advice
✅ Interview preparation strategies

Please let me know what specific area you'd like help with, and I'll provide detailed guidance!
"""