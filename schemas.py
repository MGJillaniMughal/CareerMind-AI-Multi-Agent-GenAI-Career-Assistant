"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Data Schemas with Comprehensive Validation and Types
"""

from typing import Literal, Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, validator, EmailStr
from enum import Enum
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Enumeration of available agent types"""
    RESUME_ANALYZER = "ResumeAnalyzer"
    COVER_LETTER_GENERATOR = "CoverLetterGenerator"
    JOB_SEARCHER = "JobSearcher"
    WEB_RESEARCHER = "WebResearcher"
    CAREER_ADVISOR = "CareerAdvisor"
    MARKET_ANALYST = "MarketAnalyst"
    CHATBOT = "ChatBot"
    FINISH = "Finish"


class EmploymentType(str, Enum):
    """Employment types for job searches"""
    FULL_TIME = "full-time"
    CONTRACT = "contract"
    PART_TIME = "part-time"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    VOLUNTEER = "volunteer"
    OTHER = "other"


class JobType(str, Enum):
    """Job location types"""
    ONSITE = "onsite"
    REMOTE = "remote"
    HYBRID = "hybrid"


class ExperienceLevel(str, Enum):
    """Experience levels for job filtering"""
    INTERNSHIP = "internship"
    ENTRY_LEVEL = "entry-level"
    ASSOCIATE = "associate"
    MID_SENIOR_LEVEL = "mid-senior-level"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class Priority(str, Enum):
    """Priority levels for tasks and recommendations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RouteSchema(BaseModel):
    """Enhanced routing schema for supervisor decisions"""
    next_action: Literal[
        "ResumeAnalyzer",
        "CoverLetterGenerator",
        "JobSearcher",
        "WebResearcher",
        "CareerAdvisor",
        "MarketAnalyst",
        "ChatBot",
        "Finish",
    ] = Field(
        ...,
        title="Next Action",
        description="Select the next agent to handle the user request",
    )
    
    confidence_score: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in routing decision (0.0 to 1.0)"
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation for the routing decision"
    )

    class Config:
        use_enum_values = True


class JobSearchInput(BaseModel):
    """Enhanced job search input schema with comprehensive filtering"""
    keywords: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Keywords describing the job role. Include company name if searching for specific company positions"
    )
    
    location_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description='Name of the location to search within. Example: "San Francisco, CA" or "New York City, NY"'
    )
    
    employment_type: Optional[List[EmploymentType]] = Field(
        default=None,
        description="Specific type(s) of employment to search for"
    )
    
    limit: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of jobs to retrieve (1-50)"
    )
    
    job_type: Optional[List[JobType]] = Field(
        default=None,
        description="Filter for remote, onsite, or hybrid positions"
    )
    
    experience: Optional[List[ExperienceLevel]] = Field(
        default=None,
        description='Filter by experience levels'
    )
    
    listed_at: Optional[Union[int, str]] = Field(
        default=86400,
        description="Maximum number of seconds since job posting. 86400 = last 24 hours"
    )
    
    distance: Optional[Union[int, str]] = Field(
        default=25,
        description="Maximum distance from location in miles"
    )
    
    salary_min: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum salary requirement"
    )
    
    salary_max: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum salary range"
    )
    
    company_size: Optional[str] = Field(
        default=None,
        description="Preferred company size (startup, small, medium, large, enterprise)"
    )

    @validator('salary_max')
    def validate_salary_range(cls, v, values):
        """Validate that max salary is greater than min salary"""
        if v is not None and 'salary_min' in values and values['salary_min'] is not None:
            if v < values['salary_min']:
                raise ValueError('Maximum salary must be greater than minimum salary')
        return v

    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate keywords are meaningful"""
        if not v or v.strip() == "":
            raise ValueError('Keywords cannot be empty')
        return v.strip()


class ResumeAnalysisInput(BaseModel):
    """Input schema for resume analysis requests"""
    file_path: str = Field(
        ...,
        description="Path to the resume file for analysis"
    )
    
    analysis_depth: Optional[str] = Field(
        default="comprehensive",
        description="Depth of analysis: basic, standard, comprehensive, executive"
    )
    
    target_roles: Optional[List[str]] = Field(
        default=None,
        description="Specific roles to analyze compatibility for"
    )
    
    target_industries: Optional[List[str]] = Field(
        default=None,
        description="Industries to focus analysis on"
    )
    
    include_ats_analysis: Optional[bool] = Field(
        default=True,
        description="Include ATS compatibility analysis"
    )
    
    include_market_analysis: Optional[bool] = Field(
        default=True,
        description="Include market positioning analysis"
    )


class CoverLetterInput(BaseModel):
    """Input schema for cover letter generation"""
    job_title: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Title of the position applying for"
    )
    
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the target company"
    )
    
    job_description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Full job description or key requirements"
    )
    
    resume_content: Optional[str] = Field(
        default=None,
        description="Resume content for personalization"
    )
    
    template_style: Optional[str] = Field(
        default="professional",
        description="Cover letter style: professional, creative, technical, executive"
    )
    
    tone: Optional[str] = Field(
        default="professional",
        description="Communication tone: professional, enthusiastic, formal, conversational"
    )
    
    key_achievements: Optional[List[str]] = Field(
        default=None,
        description="Specific achievements to highlight"
    )
    
    custom_requirements: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Any custom requirements or preferences"
    )


class ResearchInput(BaseModel):
    """Input schema for web research requests"""
    research_topic: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Main topic or question for research"
    )
    
    research_depth: Optional[str] = Field(
        default="standard",
        description="Research depth: quick, standard, comprehensive, deep_dive"
    )
    
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus research on"
    )
    
    time_frame: Optional[str] = Field(
        default="recent",
        description="Time frame for information: recent, last_year, historical, all"
    )
    
    source_types: Optional[List[str]] = Field(
        default=None,
        description="Preferred source types: news, academic, industry_reports, company_sites"
    )
    
    geographic_focus: Optional[str] = Field(
        default=None,
        description="Geographic focus for research"
    )


class CareerAdviceInput(BaseModel):
    """Input schema for career advisory requests"""
    current_role: Optional[str] = Field(
        default=None,
        description="Current job title or role"
    )
    
    experience_years: Optional[int] = Field(
        default=None,
        ge=0,
        le=50,
        description="Years of professional experience"
    )
    
    current_industry: Optional[str] = Field(
        default=None,
        description="Current industry or sector"
    )
    
    target_role: Optional[str] = Field(
        default=None,
        description="Desired future role or position"
    )
    
    target_industry: Optional[str] = Field(
        default=None,
        description="Target industry for career transition"
    )
    
    career_goals: Optional[List[str]] = Field(
        default=None,
        description="Specific career goals and aspirations"
    )
    
    timeline: Optional[str] = Field(
        default=None,
        description="Timeline for career goals (6 months, 1 year, 2-5 years, long-term)"
    )
    
    challenges: Optional[List[str]] = Field(
        default=None,
        description="Current career challenges or concerns"
    )
    
    strengths: Optional[List[str]] = Field(
        default=None,
        description="Known professional strengths"
    )
    
    skills_to_develop: Optional[List[str]] = Field(
        default=None,
        description="Skills interested in developing"
    )


class MarketAnalysisInput(BaseModel):
    """Input schema for market analysis requests"""
    analysis_type: str = Field(
        ...,
        description="Type of analysis: industry_trends, salary_analysis, job_market, competitive_landscape"
    )
    
    target_industry: Optional[str] = Field(
        default=None,
        description="Industry to analyze"
    )
    
    target_roles: Optional[List[str]] = Field(
        default=None,
        description="Specific roles to include in analysis"
    )
    
    geographic_scope: Optional[str] = Field(
        default="global",
        description="Geographic scope: local, national, regional, global"
    )
    
    time_horizon: Optional[str] = Field(
        default="current",
        description="Time horizon: current, 6_months, 1_year, 3_years, 5_years"
    )
    
    data_sources: Optional[List[str]] = Field(
        default=None,
        description="Preferred data sources for analysis"
    )
    
    metrics_focus: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to focus on"
    )


class UserPreferences(BaseModel):
    """User preferences and settings"""
    preferred_industries: Optional[List[str]] = Field(
        default=None,
        description="User's preferred industries"
    )
    
    location_preferences: Optional[List[str]] = Field(
        default=None,
        description="Preferred work locations"
    )
    
    remote_preference: Optional[str] = Field(
        default="flexible",
        description="Remote work preference: remote_only, hybrid, onsite, flexible"
    )
    
    salary_expectations: Optional[Dict[str, int]] = Field(
        default=None,
        description="Salary expectations with min/max values"
    )
    
    communication_style: Optional[str] = Field(
        default="professional",
        description="Preferred communication style"
    )
    
    notification_preferences: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Notification preferences for different types of updates"
    )


class SessionState(BaseModel):
    """Enhanced session state management"""
    session_id: str = Field(
        ...,
        description="Unique session identifier"
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier if authenticated"
    )
    
    current_agent: Optional[str] = Field(
        default=None,
        description="Currently active agent"
    )
    
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history"
    )
    
    user_preferences: Optional[UserPreferences] = Field(
        default=None,
        description="User preferences and settings"
    )
    
    resume_uploaded: bool = Field(
        default=False,
        description="Whether user has uploaded a resume"
    )
    
    analysis_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stored analysis results"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Session creation timestamp"
    )
    
    last_activity: datetime = Field(
        default_factory=datetime.now,
        description="Last activity timestamp"
    )


class JobResult(BaseModel):
    """Schema for job search results"""
    job_title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    job_description: Optional[str] = Field(default=None, description="Job description")
    apply_url: Optional[str] = Field(default=None, description="Application URL")
    salary_range: Optional[str] = Field(default=None, description="Salary range")
    posted_date: Optional[str] = Field(default=None, description="Date posted")
    employment_type: Optional[str] = Field(default=None, description="Employment type")
    experience_level: Optional[str] = Field(default=None, description="Required experience level")
    remote_option: Optional[bool] = Field(default=None, description="Remote work availability")
    company_size: Optional[str] = Field(default=None, description="Company size")
    industry: Optional[str] = Field(default=None, description="Company industry")
    benefits: Optional[List[str]] = Field(default=None, description="Benefits offered")
    skills_required: Optional[List[str]] = Field(default=None, description="Required skills")
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Relevance score")


class ResumeAnalysisResult(BaseModel):
    """Schema for resume analysis results"""
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall resume score")
    
    skills_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Skills analysis results"
    )
    
    experience_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Experience analysis results"
    )
    
    education_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Education analysis results"
    )
    
    ats_compatibility: Dict[str, Any] = Field(
        default_factory=dict,
        description="ATS compatibility analysis"
    )
    
    market_positioning: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market positioning analysis"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    
    target_role_fit: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fit scores for target roles"
    )


class CoverLetterResult(BaseModel):
    """Schema for cover letter generation results"""
    cover_letter_content: str = Field(..., description="Generated cover letter content")
    file_path: str = Field(..., description="Path to generated file")
    template_used: str = Field(..., description="Template style used")
    personalization_score: float = Field(..., ge=0.0, le=1.0, description="Personalization quality score")
    ats_optimized: bool = Field(..., description="Whether content is ATS optimized")
    word_count: int = Field(..., description="Word count of the letter")
    key_highlights: List[str] = Field(default_factory=list, description="Key points highlighted")


class ResearchResult(BaseModel):
    """Schema for research results"""
    research_summary: str = Field(..., description="Executive summary of research")
    key_findings: List[str] = Field(default_factory=list, description="Key research findings")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Research sources")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Data quality assessment")
    research_depth: str = Field(..., description="Actual research depth achieved")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    agent: Optional[str] = Field(default=None, description="Agent where error occurred")
    user_message: str = Field(..., description="User-friendly error message")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested user actions")


class SuccessResponse(BaseModel):
    """Schema for successful responses"""
    success: bool = Field(default=True, description="Success indicator")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    agent: Optional[str] = Field(default=None, description="Agent that processed request")
    next_steps: Optional[List[str]] = Field(default=None, description="Suggested next steps")


# Validation utilities
def validate_email(email: str) -> bool:
    """Validate email format"""
    try:
        EmailStr.validate(email)
        return True
    except:
        return False


def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    import re
    phone_pattern = re.compile(r'^\+?1?\d{9,15}$')
    return bool(phone_pattern.match(phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')))


def validate_url(url: str) -> bool:
    """Validate URL format"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))


# Schema factory functions
def create_job_search_schema(**kwargs) -> JobSearchInput:
    """Factory function to create job search input with validation"""
    try:
        return JobSearchInput(**kwargs)
    except Exception as e:
        logger.error(f"Error creating job search schema: {str(e)}")
        raise ValueError(f"Invalid job search parameters: {str(e)}")


def create_resume_analysis_schema(**kwargs) -> ResumeAnalysisInput:
    """Factory function to create resume analysis input with validation"""
    try:
        return ResumeAnalysisInput(**kwargs)
    except Exception as e:
        logger.error(f"Error creating resume analysis schema: {str(e)}")
        raise ValueError(f"Invalid resume analysis parameters: {str(e)}")


def create_cover_letter_schema(**kwargs) -> CoverLetterInput:
    """Factory function to create cover letter input with validation"""
    try:
        return CoverLetterInput(**kwargs)
    except Exception as e:
        logger.error(f"Error creating cover letter schema: {str(e)}")
        raise ValueError(f"Invalid cover letter parameters: {str(e)}")


# Default configurations
DEFAULT_JOB_SEARCH_CONFIG = {
    "limit": 10,
    "listed_at": 86400,  # Last 24 hours
    "distance": 25,  # 25 miles
    "employment_type": [EmploymentType.FULL_TIME],
    "job_type": [JobType.ONSITE, JobType.REMOTE, JobType.HYBRID]
}

DEFAULT_RESUME_ANALYSIS_CONFIG = {
    "analysis_depth": "comprehensive",
    "include_ats_analysis": True,
    "include_market_analysis": True
}

DEFAULT_COVER_LETTER_CONFIG = {
    "template_style": "professional",
    "tone": "professional"
}

DEFAULT_RESEARCH_CONFIG = {
    "research_depth": "standard",
    "time_frame": "recent"
}

DEFAULT_USER_PREFERENCES = {
    "remote_preference": "flexible",
    "communication_style": "professional"
}