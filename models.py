from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ApplicationStatus(str, Enum):
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    email: EmailStr
    name: str
    role: UserRole = UserRole.USER
    xp: int = 0
    level: int = 1
    streak: int = 0
    last_activity: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: User

class Problem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    title: str
    difficulty: DifficultyLevel
    topics: List[str]
    platform: str
    solved_at: datetime
    time_taken: Optional[int] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProblemCreate(BaseModel):
    title: str
    difficulty: DifficultyLevel
    topics: List[str]
    platform: str = "leetcode"
    time_taken: Optional[int] = None
    notes: Optional[str] = None

class Subject(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    name: str
    category: str
    mastery_score: float = 0.0
    last_practiced: Optional[datetime] = None
    total_problems: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Company(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    name: str
    target_role: str
    required_skills: List[str]
    difficulty_level: DifficultyLevel
    readiness_score: float = 0.0
    skill_gaps: List[str] = []
    priority: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CompanyCreate(BaseModel):
    name: str
    target_role: str
    required_skills: List[str]
    difficulty_level: DifficultyLevel

class Application(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    company_name: str
    position: str
    status: ApplicationStatus
    applied_date: datetime
    interview_date: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ApplicationCreate(BaseModel):
    company_name: str
    position: str
    status: ApplicationStatus = ApplicationStatus.APPLIED
    applied_date: Optional[datetime] = None
    interview_date: Optional[datetime] = None
    notes: Optional[str] = None

class ApplicationUpdate(BaseModel):
    status: Optional[ApplicationStatus] = None
    interview_date: Optional[datetime] = None
    notes: Optional[str] = None

class MockInterview(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    company_name: str
    role: str
    difficulty: DifficultyLevel
    questions: List[Dict[str, Any]]
    answers: List[Dict[str, Any]] = []
    overall_score: Optional[float] = None
    feedback: Optional[str] = None
    completed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MockInterviewCreate(BaseModel):
    company_name: str
    role: str
    difficulty: DifficultyLevel

class MockInterviewAnswer(BaseModel):
    question_id: int
    answer: str

class ResumeAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    filename: str
    extracted_text: str
    ats_score: float
    keyword_gaps: List[str]
    missing_skills: List[str]
    suggestions: List[str]
    company_specific: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    session_id: str
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ReadinessMetric(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str
    user_id: str
    overall_readiness: float
    weak_topics: List[str]
    strong_topics: List[str]
    interview_probability: float
    application_conversion_rate: float
    skill_velocity: float
    next_action: str
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))