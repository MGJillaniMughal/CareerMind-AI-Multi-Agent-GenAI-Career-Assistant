"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Team Members Configuration with Detailed Roles and Capabilities
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentTier(Enum):
    """Agent performance and capability tiers"""
    EXPERT = "expert"
    SPECIALIST = "specialist"
    ASSISTANT = "assistant"
    COORDINATOR = "coordinator"


class AgentCapability(Enum):
    """Specific capabilities that agents possess"""
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_RESEARCH = "web_research"
    CONTENT_GENERATION = "content_generation"
    JOB_SEARCH = "job_search"
    CAREER_GUIDANCE = "career_guidance"
    MARKET_ANALYSIS = "market_analysis"
    CONVERSATION = "conversation"
    COORDINATION = "coordination"


@dataclass
class AgentProfile:
    """Comprehensive agent profile with enhanced metadata"""
    name: str
    description: str
    specialization: str
    tier: AgentTier
    capabilities: List[AgentCapability]
    use_cases: List[str]
    tools: List[str]
    performance_metrics: Dict[str, Any]
    interaction_style: str
    limitations: List[str]
    ideal_scenarios: List[str]


class EnhancedTeamManager:
    """
    Enhanced team manager for organizing and managing AI agents.
    """
    
    def __init__(self):
        """Initialize the enhanced team manager"""
        self.agent_profiles = self._initialize_agent_profiles()
        self.team_hierarchy = self._build_team_hierarchy()
        
        logger.info("Enhanced Team Manager initialized with comprehensive agent profiles")
    
    def _initialize_agent_profiles(self) -> Dict[str, AgentProfile]:
        """
        Initialize comprehensive profiles for all team members.
        
        Returns:
            Dict mapping agent names to their detailed profiles
        """
        return {
            "ResumeAnalyzer": AgentProfile(
                name="ResumeAnalyzer",
                description="Advanced resume analysis specialist with deep document parsing and professional assessment capabilities",
                specialization="Professional Document Analysis & Career Assessment",
                tier=AgentTier.EXPERT,
                capabilities=[
                    AgentCapability.DOCUMENT_ANALYSIS,
                    AgentCapability.CAREER_GUIDANCE
                ],
                use_cases=[
                    "Comprehensive resume analysis and scoring",
                    "Skills extraction and categorization",
                    "Experience timeline analysis",
                    "ATS compatibility assessment",
                    "Professional strengths identification",
                    "Career progression mapping",
                    "Resume improvement recommendations"
                ],
                tools=[
                    "ResumeExtractorTool",
                    "career_path_analyzer",
                    "skills_assessment_tool"
                ],
                performance_metrics={
                    "accuracy": 95,
                    "processing_speed": "fast",
                    "detail_level": "comprehensive",
                    "user_satisfaction": 4.8
                },
                interaction_style="Professional, detailed, and constructive",
                limitations=[
                    "Requires well-formatted resume documents",
                    "Cannot analyze handwritten resumes",
                    "Limited to standard resume formats"
                ],
                ideal_scenarios=[
                    "Professional resume review and feedback",
                    "Career transition planning",
                    "Job application preparation",
                    "Skills gap analysis"
                ]
            ),
            
            "JobSearcher": AgentProfile(
                name="JobSearcher",
                description="Intelligent job discovery specialist with advanced search algorithms and market intelligence",
                specialization="Job Market Research & Opportunity Discovery",
                tier=AgentTier.EXPERT,
                capabilities=[
                    AgentCapability.JOB_SEARCH,
                    AgentCapability.WEB_RESEARCH,
                    AgentCapability.MARKET_ANALYSIS
                ],
                use_cases=[
                    "Targeted job opportunity discovery",
                    "Company-specific position searches",
                    "Salary range analysis",
                    "Job market trend identification",
                    "Remote work opportunity finding",
                    "Industry-specific job filtering",
                    "Application deadline tracking"
                ],
                tools=[
                    "get_job_search_tool",
                    "salary_analyzer_tool",
                    "company_research_tool"
                ],
                performance_metrics={
                    "job_relevance": 92,
                    "search_coverage": "comprehensive",
                    "update_frequency": "real-time",
                    "success_rate": 4.7
                },
                interaction_style="Efficient, thorough, and results-oriented",
                limitations=[
                    "Dependent on job board availability",
                    "May miss unlisted positions",
                    "Limited to English-language job posts"
                ],
                ideal_scenarios=[
                    "Active job searching",
                    "Market research for career planning",
                    "Salary benchmarking",
                    "Company opportunity assessment"
                ]
            ),
            
            "CoverLetterGenerator": AgentProfile(
                name="CoverLetterGenerator",
                description="Expert content creation specialist for personalized application materials and professional communication",
                specialization="Professional Content Creation & Application Optimization",
                tier=AgentTier.EXPERT,
                capabilities=[
                    AgentCapability.CONTENT_GENERATION,
                    AgentCapability.DOCUMENT_ANALYSIS
                ],
                use_cases=[
                    "Personalized cover letter creation",
                    "Application email drafting",
                    "Professional communication templates",
                    "Job-specific content optimization",
                    "ATS-friendly content formatting",
                    "Multi-language support",
                    "Industry-specific customization"
                ],
                tools=[
                    "generate_letter_for_specific_job",
                    "save_cover_letter_for_specific_job",
                    "ResumeExtractorTool",
                    "content_optimizer_tool"
                ],
                performance_metrics={
                    "personalization": 94,
                    "professional_quality": 96,
                    "ats_compatibility": 90,
                    "user_satisfaction": 4.9
                },
                interaction_style="Creative, professional, and persuasive",
                limitations=[
                    "Requires detailed job descriptions for best results",
                    "Limited to standard business communication styles",
                    "Cannot guarantee job application success"
                ],
                ideal_scenarios=[
                    "Job application preparation",
                    "Professional networking communications",
                    "Career transition letters",
                    "Follow-up correspondence"
                ]
            ),
            
            "WebResearcher": AgentProfile(
                name="WebResearcher",
                description="Advanced web intelligence specialist with comprehensive research and analysis capabilities",
                specialization="Information Gathering & Market Intelligence",
                tier=AgentTier.SPECIALIST,
                capabilities=[
                    AgentCapability.WEB_RESEARCH,
                    AgentCapability.MARKET_ANALYSIS,
                    AgentCapability.CONTENT_GENERATION
                ],
                use_cases=[
                    "Company background research",
                    "Industry trend analysis",
                    "Competitive intelligence gathering",
                    "News and market updates",
                    "Technology landscape research",
                    "Professional networking insights",
                    "Regulatory and compliance information"
                ],
                tools=[
                    "get_google_search_results",
                    "scrape_website",
                    "analyze_job_market_trends",
                    "company_intelligence_tool"
                ],
                performance_metrics={
                    "information_accuracy": 93,
                    "research_depth": "comprehensive",
                    "source_reliability": 91,
                    "processing_speed": "fast"
                },
                interaction_style="Analytical, thorough, and objective",
                limitations=[
                    "Dependent on publicly available information",
                    "Cannot access private databases",
                    "Limited by website accessibility"
                ],
                ideal_scenarios=[
                    "Interview preparation research",
                    "Company due diligence",
                    "Industry analysis",
                    "Competitive landscape mapping"
                ]
            ),
            
            "CareerAdvisor": AgentProfile(
                name="CareerAdvisor",
                description="Strategic career development specialist with personalized guidance and long-term planning expertise",
                specialization="Career Strategy & Professional Development",
                tier=AgentTier.EXPERT,
                capabilities=[
                    AgentCapability.CAREER_GUIDANCE,
                    AgentCapability.DOCUMENT_ANALYSIS,
                    AgentCapability.MARKET_ANALYSIS
                ],
                use_cases=[
                    "Career path planning and strategy",
                    "Professional development roadmaps",
                    "Skills gap analysis and recommendations",
                    "Industry transition guidance",
                    "Leadership development planning",
                    "Networking strategy development",
                    "Personal branding consultation"
                ],
                tools=[
                    "ResumeExtractorTool",
                    "career_path_analyzer",
                    "skills_gap_analyzer",
                    "professional_development_planner"
                ],
                performance_metrics={
                    "guidance_quality": 96,
                    "personalization": 95,
                    "actionability": 93,
                    "long_term_success": 4.8
                },
                interaction_style="Supportive, strategic, and motivational",
                limitations=[
                    "Cannot guarantee career outcomes",
                    "Advice based on general market trends",
                    "Limited to publicly available career data"
                ],
                ideal_scenarios=[
                    "Career transition planning",
                    "Professional development strategy",
                    "Long-term career goal setting",
                    "Industry change preparation"
                ]
            ),
            
            "MarketAnalyst": AgentProfile(
                name="MarketAnalyst",
                description="Data-driven market intelligence specialist with advanced analytical capabilities for industry insights",
                specialization="Market Research & Economic Analysis",
                tier=AgentTier.EXPERT,
                capabilities=[
                    AgentCapability.MARKET_ANALYSIS,
                    AgentCapability.WEB_RESEARCH,
                    AgentCapability.CONTENT_GENERATION
                ],
                use_cases=[
                    "Industry trend analysis and forecasting",
                    "Salary benchmarking and compensation analysis",
                    "Job market demand assessment",
                    "Economic impact analysis",
                    "Skill demand forecasting",
                    "Geographic market comparison",
                    "Startup ecosystem analysis"
                ],
                tools=[
                    "get_google_search_results",
                    "analyze_job_market_trends",
                    "salary_analyzer_tool",
                    "economic_data_analyzer"
                ],
                performance_metrics={
                    "analytical_accuracy": 94,
                    "trend_prediction": 89,
                    "data_comprehensiveness": 92,
                    "insight_quality": 4.7
                },
                interaction_style="Data-driven, analytical, and insightful",
                limitations=[
                    "Predictions based on historical data",
                    "Cannot account for unpredictable market events",
                    "Limited to publicly available market data"
                ],
                ideal_scenarios=[
                    "Market entry planning",
                    "Investment decision support",
                    "Business strategy development",
                    "Competitive analysis"
                ]
            ),
            
            "ChatBot": AgentProfile(
                name="ChatBot",
                description="Intelligent conversational assistant with natural language processing and context-aware responses",
                specialization="Conversational AI & General Assistance",
                tier=AgentTier.ASSISTANT,
                capabilities=[
                    AgentCapability.CONVERSATION,
                    AgentCapability.CONTENT_GENERATION
                ],
                use_cases=[
                    "General Q&A and clarification",
                    "Conversation flow management",
                    "Information formatting and presentation",
                    "Follow-up question handling",
                    "Context-aware responses",
                    "User guidance and navigation",
                    "Session summarization"
                ],
                tools=[
                    "natural_language_processor",
                    "context_manager",
                    "response_formatter"
                ],
                performance_metrics={
                    "response_quality": 88,
                    "context_awareness": 92,
                    "user_satisfaction": 4.5,
                    "response_speed": "instant"
                },
                interaction_style="Friendly, helpful, and conversational",
                limitations=[
                    "Limited specialized knowledge",
                    "Cannot perform complex analysis",
                    "Dependent on conversation context"
                ],
                ideal_scenarios=[
                    "General inquiries and clarifications",
                    "Conversation management",
                    "Information formatting",
                    "User assistance and guidance"
                ]
            ),
            
            "Supervisor": AgentProfile(
                name="Supervisor",
                description="Intelligent routing and coordination specialist managing multi-agent workflows and decision-making",
                specialization="Workflow Orchestration & Agent Coordination",
                tier=AgentTier.COORDINATOR,
                capabilities=[
                    AgentCapability.COORDINATION,
                    AgentCapability.CONVERSATION
                ],
                use_cases=[
                    "Agent routing and task delegation",
                    "Workflow optimization",
                    "Context analysis and understanding",
                    "Multi-agent coordination",
                    "Quality assurance and validation",
                    "Error handling and recovery",
                    "Performance monitoring"
                ],
                tools=[
                    "routing_engine",
                    "context_analyzer",
                    "workflow_optimizer",
                    "quality_validator"
                ],
                performance_metrics={
                    "routing_accuracy": 96,
                    "workflow_efficiency": 94,
                    "decision_speed": "fast",
                    "system_reliability": 4.9
                },
                interaction_style="Efficient, decisive, and systematic",
                limitations=[
                    "Dependent on agent availability",
                    "Cannot create new capabilities",
                    "Limited to predefined workflows"
                ],
                ideal_scenarios=[
                    "Complex multi-step tasks",
                    "Workflow management",
                    "Agent coordination",
                    "System optimization"
                ]
            )
        }
    
    def _build_team_hierarchy(self) -> Dict[str, List[str]]:
        """
        Build team hierarchy and relationships.
        
        Returns:
            Dict defining team structure and relationships
        """
        return {
            "coordinators": ["Supervisor"],
            "experts": ["ResumeAnalyzer", "JobSearcher", "CoverLetterGenerator", "CareerAdvisor", "MarketAnalyst"],
            "specialists": ["WebResearcher"],
            "assistants": ["ChatBot"],
            "document_processors": ["ResumeAnalyzer", "CoverLetterGenerator"],
            "researchers": ["WebResearcher", "JobSearcher", "MarketAnalyst"],
            "advisors": ["CareerAdvisor", "MarketAnalyst"],
            "content_creators": ["CoverLetterGenerator", "ChatBot"]
        }
    
    def get_team_members_details(self) -> List[Dict[str, Any]]:
        """
        Get comprehensive team member details in the original format for compatibility.
        
        Returns:
            List of team member dictionaries
        """
        try:
            members_list = []
            
            for agent_name, profile in self.agent_profiles.items():
                if agent_name != "Supervisor":  # Exclude supervisor from worker list
                    member_dict = {
                        "name": profile.name,
                        "description": profile.description,
                        "specialization": profile.specialization,
                        "use_cases": "; ".join(profile.use_cases[:3]),  # Top 3 use cases
                        "tier": profile.tier.value,
                        "capabilities": [cap.value for cap in profile.capabilities]
                    }
                    members_list.append(member_dict)
            
            # Add Finish option
            members_list.append({
                "name": "Finish",
                "description": "Completes the current workflow and provides final results to the user",
                "specialization": "Workflow Completion",
                "use_cases": "Task completion; Final response delivery; Session closure",
                "tier": "system",
                "capabilities": ["workflow_completion"]
            })
            
            logger.info(f"Retrieved {len(members_list)} team member details")
            return members_list
            
        except Exception as e:
            logger.error(f"Error getting team member details: {str(e)}")
            return self._get_fallback_members()
    
    def get_agent_profile(self, agent_name: str) -> AgentProfile:
        """
        Get detailed profile for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentProfile: Detailed agent profile
        """
        try:
            profile = self.agent_profiles.get(agent_name)
            if not profile:
                logger.warning(f"Profile not found for agent: {agent_name}")
                return self._get_default_profile(agent_name)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting agent profile for {agent_name}: {str(e)}")
            return self._get_default_profile(agent_name)
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[str]:
        """
        Get agents that possess a specific capability.
        
        Args:
            capability: Desired capability
            
        Returns:
            List of agent names with the capability
        """
        try:
            capable_agents = []
            
            for agent_name, profile in self.agent_profiles.items():
                if capability in profile.capabilities:
                    capable_agents.append(agent_name)
            
            return capable_agents
            
        except Exception as e:
            logger.error(f"Error getting agents by capability: {str(e)}")
            return []
    
    def get_recommended_agent(self, task_description: str) -> str:
        """
        Get recommended agent based on task description.
        
        Args:
            task_description: Description of the task to be performed
            
        Returns:
            str: Recommended agent name
        """
        try:
            task_lower = task_description.lower()
            
            # Task-based routing logic
            if any(keyword in task_lower for keyword in ["resume", "cv", "analyze", "skills", "experience"]):
                return "ResumeAnalyzer"
            elif any(keyword in task_lower for keyword in ["job", "search", "position", "hiring", "opportunities"]):
                return "JobSearcher"
            elif any(keyword in task_lower for keyword in ["cover letter", "application", "letter", "write"]):
                return "CoverLetterGenerator"
            elif any(keyword in task_lower for keyword in ["research", "company", "information", "trends"]):
                return "WebResearcher"
            elif any(keyword in task_lower for keyword in ["career", "advice", "guidance", "plan", "development"]):
                return "CareerAdvisor"
            elif any(keyword in task_lower for keyword in ["market", "salary", "industry", "analysis", "trends"]):
                return "MarketAnalyst"
            else:
                return "ChatBot"
                
        except Exception as e:
            logger.error(f"Error getting recommended agent: {str(e)}")
            return "ChatBot"
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive team statistics and capabilities overview.
        
        Returns:
            Dict containing team statistics
        """
        try:
            total_agents = len(self.agent_profiles)
            tier_distribution = {}
            capability_distribution = {}
            
            for profile in self.agent_profiles.values():
                # Count tiers
                tier = profile.tier.value
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
                
                # Count capabilities
                for capability in profile.capabilities:
                    cap_name = capability.value
                    capability_distribution[cap_name] = capability_distribution.get(cap_name, 0) + 1
            
            return {
                "total_agents": total_agents,
                "tier_distribution": tier_distribution,
                "capability_distribution": capability_distribution,
                "team_hierarchy": self.team_hierarchy,
                "specialized_agents": len([p for p in self.agent_profiles.values() if p.tier != AgentTier.ASSISTANT])
            }
            
        except Exception as e:
            logger.error(f"Error getting team statistics: {str(e)}")
            return {}
    
    def _get_fallback_members(self) -> List[Dict[str, Any]]:
        """
        Get fallback member list in case of errors.
        
        Returns:
            List of basic team member dictionaries
        """
        return [
            {
                "name": "ResumeAnalyzer",
                "description": "Analyzes resumes and extracts key information for career assessment",
                "specialization": "Resume Analysis",
                "use_cases": "Resume review; Skills extraction; Career assessment"
            },
            {
                "name": "JobSearcher",
                "description": "Searches for job opportunities based on specified criteria and preferences",
                "specialization": "Job Discovery",
                "use_cases": "Job searching; Opportunity discovery; Market research"
            },
            {
                "name": "CoverLetterGenerator",
                "description": "Creates personalized cover letters tailored to specific job applications",
                "specialization": "Content Creation",
                "use_cases": "Cover letter writing; Application materials; Professional communication"
            },
            {
                "name": "WebResearcher",
                "description": "Conducts comprehensive web research to gather relevant information",
                "specialization": "Information Gathering",
                "use_cases": "Web research; Company information; Industry insights"
            },
            {
                "name": "ChatBot",
                "description": "Provides general assistance and handles conversational interactions",
                "specialization": "General Assistance",
                "use_cases": "Q&A; General help; Conversation management"
            },
            {
                "name": "Finish",
                "description": "Completes the workflow and provides final results",
                "specialization": "Workflow Completion",
                "use_cases": "Task completion; Final responses; Session closure"
            }
        ]
    
    def _get_default_profile(self, agent_name: str) -> AgentProfile:
        """
        Get a default profile for unknown agents.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentProfile: Default profile
        """
        return AgentProfile(
            name=agent_name,
            description=f"AI assistant specializing in {agent_name.lower()} tasks",
            specialization="General AI Assistant",
            tier=AgentTier.ASSISTANT,
            capabilities=[AgentCapability.CONVERSATION],
            use_cases=["General assistance"],
            tools=["basic_tools"],
            performance_metrics={"accuracy": 85},
            interaction_style="Helpful and professional",
            limitations=["Limited specialized knowledge"],
            ideal_scenarios=["General inquiries"]
        )


# Global team manager instance
team_manager = EnhancedTeamManager()


# Backward compatibility functions
def get_team_members_details() -> List[Dict[str, Any]]:
    """
    Backward compatible function to get team member details.
    
    Returns:
        List of team member dictionaries
    """
    try:
        return team_manager.get_team_members_details()
    except Exception as e:
        logger.error(f"Error in backward compatible get_team_members_details: {str(e)}")
        return team_manager._get_fallback_members()


def get_enhanced_team_members_details() -> List[Dict[str, Any]]:
    """
    Enhanced function to get comprehensive team member details.
    
    Returns:
        List of enhanced team member dictionaries
    """
    try:
        return team_manager.get_team_members_details()
    except Exception as e:
        logger.error(f"Error getting enhanced team member details: {str(e)}")
        return team_manager._get_fallback_members()


# Utility functions
def get_agent_capabilities() -> Dict[str, List[str]]:
    """
    Get a mapping of agents to their capabilities.
    
    Returns:
        Dict mapping agent names to capability lists
    """
    try:
        capabilities_map = {}
        
        for agent_name, profile in team_manager.agent_profiles.items():
            capabilities_map[agent_name] = [cap.value for cap in profile.capabilities]
        
        return capabilities_map
        
    except Exception as e:
        logger.error(f"Error getting agent capabilities: {str(e)}")
        return {}


def get_workflow_recommendations(task_complexity: str = "simple") -> List[str]:
    """
    Get recommended workflow based on task complexity.
    
    Args:
        task_complexity: Complexity level (simple, moderate, complex)
        
    Returns:
        List of recommended agent sequence
    """
    workflows = {
        "simple": ["ChatBot"],
        "moderate": ["ResumeAnalyzer", "JobSearcher"],
        "complex": ["ResumeAnalyzer", "WebResearcher", "CareerAdvisor", "JobSearcher", "CoverLetterGenerator"]
    }
    
    return workflows.get(task_complexity, ["ChatBot"])


def validate_agent_availability(agent_names: List[str]) -> Dict[str, bool]:
    """
    Validate which agents are available and properly configured.
    
    Args:
        agent_names: List of agent names to validate
        
    Returns:
        Dict mapping agent names to availability status
    """
    try:
        availability = {}
        
        for agent_name in agent_names:
            # Check if agent exists in profiles
            availability[agent_name] = agent_name in team_manager.agent_profiles
        
        return availability
        
    except Exception as e:
        logger.error(f"Error validating agent availability: {str(e)}")
        return {agent: False for agent in agent_names}