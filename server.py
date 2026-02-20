from fastapi import FastAPI, APIRouter, HTTPException, status, Depends, File, UploadFile, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Optional
import uuid
from datetime import datetime, timezone

# Uvicorn is sometimes launched from inside the `backend/` folder (e.g. `uvicorn server:app`).
# In that case, `backend.*` absolute imports fail unless the project root is on sys.path.
import sys


def _ensure_project_root_on_syspath() -> None:
    try:
        import backend  # noqa: F401
        return
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_syspath()

from backend.models import (  # noqa: E402
    User,
    UserCreate,
    UserLogin,
    Token,
    Problem,
    ProblemCreate,
    Company,
    CompanyCreate,
    Application,
    ApplicationCreate,
    ApplicationUpdate,
    MockInterview,
    MockInterviewCreate,
    MockInterviewAnswer,
    ResumeAnalysis,
    ChatMessage,
    ChatRequest,
    ReadinessMetric,
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# IMPORTANT: modules below read env vars at import-time.
from backend.auth_service import (  # noqa: E402
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from backend.ai_service import ai_service  # noqa: E402
from backend.resume_service import resume_service  # noqa: E402
from backend.recommendation_service import recommendation_service  # noqa: E402

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()

def _parse_cors_origins(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if not raw or raw == "*":
        return []
    # Supports comma-separated origins.
    return [o.strip() for o in raw.split(",") if o.strip()]


dev_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

extra_origins = _parse_cors_origins(os.environ.get("CORS_ORIGINS"))
allowed_origins = list(dict.fromkeys(dev_origins + extra_origins))

# Add CORS middleware immediately after app creation - CRITICAL for proper CORS handling
# NOTE: withCredentials=true in the frontend requires explicit origins (not '*').
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

api_router = APIRouter(prefix="/api")
security = HTTPBearer()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    user = await db.users.find_one({"id": user_id}, {"_id": 0, "hashed_password": 0})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    
    return user

@api_router.get("/")
async def root():
    return {"message": "Placement Intelligence API", "cors": "enabled"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "cors": "configured"}

@api_router.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    existing = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    user = User(
        id=user_id,
        email=user_data.email,
        name=user_data.name,
        created_at=datetime.now(timezone.utc)
    )
    
    user_doc = user.model_dump()
    user_doc["hashed_password"] = hashed_password
    user_doc["created_at"] = user_doc["created_at"].isoformat()
    
    await db.users.insert_one(user_doc)
    
    access_token = create_access_token({"sub": user_id})
    refresh_token = create_refresh_token({"sub": user_id})
    
    return Token(access_token=access_token, refresh_token=refresh_token, user=user)

@api_router.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    user_doc = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    if not verify_password(credentials.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    if isinstance(user_doc.get("created_at"), str):
        user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
    
    user_doc.pop("hashed_password", None)
    user = User(**user_doc)
    
    access_token = create_access_token({"sub": user.id})
    refresh_token = create_refresh_token({"sub": user.id})
    
    return Token(access_token=access_token, refresh_token=refresh_token, user=user)

@api_router.post("/auth/refresh", response_model=Token)
async def refresh_token(refresh_token: str = Header(...)):
    payload = decode_token(refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    
    user_id = payload.get("sub")
    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0, "hashed_password": 0})
    
    if not user_doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    
    if isinstance(user_doc.get("created_at"), str):
        user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
    
    user = User(**user_doc)
    
    access_token = create_access_token({"sub": user_id})
    new_refresh_token = create_refresh_token({"sub": user_id})
    
    return Token(access_token=access_token, refresh_token=new_refresh_token, user=user)

@api_router.get("/dashboard")
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    
    problems = await db.problems.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    for p in problems:
        if isinstance(p.get("solved_at"), str):
            p["solved_at"] = datetime.fromisoformat(p["solved_at"])
    
    applications = await db.applications.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    subjects = await db.subjects.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    
    readiness_score = recommendation_service.calculate_readiness_score(current_user, problems, subjects)
    weak_topics = recommendation_service.detect_weak_topics(problems)
    strong_topics = recommendation_service.detect_strong_topics(problems)
    interview_probability = recommendation_service.calculate_interview_probability(readiness_score, applications)
    conversion_rate = recommendation_service.calculate_application_conversion_rate(applications)
    skill_velocity = recommendation_service.calculate_skill_velocity(problems)
    next_action = recommendation_service.generate_next_action(readiness_score, weak_topics, applications)
    
    metric = ReadinessMetric(
        id=str(uuid.uuid4()),
        user_id=user_id,
        overall_readiness=readiness_score,
        weak_topics=weak_topics,
        strong_topics=strong_topics,
        interview_probability=interview_probability,
        application_conversion_rate=conversion_rate,
        skill_velocity=skill_velocity,
        next_action=next_action
    )
    
    metric_doc = metric.model_dump()
    metric_doc["calculated_at"] = metric_doc["calculated_at"].isoformat()
    await db.readiness_metrics.insert_one(metric_doc)
    
    return {
        "metrics": metric,
        "user": current_user,
        "total_problems": len(problems),
        "total_applications": len(applications),
        "recent_problems": problems[:5] if problems else []
    }

@api_router.post("/problems", response_model=Problem)
async def create_problem(problem_data: ProblemCreate, current_user: dict = Depends(get_current_user)):
    problem_id = str(uuid.uuid4())
    problem = Problem(
        id=problem_id,
        user_id=current_user["id"],
        solved_at=datetime.now(timezone.utc),
        **problem_data.model_dump()
    )
    
    doc = problem.model_dump()
    doc["solved_at"] = doc["solved_at"].isoformat()
    doc["created_at"] = doc["created_at"].isoformat()
    
    await db.problems.insert_one(doc)
    
    xp_gain = {"easy": 10, "medium": 20, "hard": 30}.get(problem_data.difficulty, 10)
    await db.users.update_one({"id": current_user["id"]}, {"$inc": {"xp": xp_gain}})
    
    return problem

@api_router.get("/problems", response_model=List[Problem])
async def get_problems(current_user: dict = Depends(get_current_user)):
    problems = await db.problems.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    for p in problems:
        if isinstance(p.get("solved_at"), str):
            p["solved_at"] = datetime.fromisoformat(p["solved_at"])
        if isinstance(p.get("created_at"), str):
            p["created_at"] = datetime.fromisoformat(p["created_at"])
    return problems

@api_router.delete("/problems/{problem_id}")
async def delete_problem(problem_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.problems.delete_one({"id": problem_id, "user_id": current_user["id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Problem not found")
    return {"message": "Problem deleted"}

@api_router.post("/companies", response_model=Company)
async def create_company(company_data: CompanyCreate, current_user: dict = Depends(get_current_user)):
    company_id = str(uuid.uuid4())
    
    user_problems = await db.problems.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    
    company_dict = company_data.model_dump()
    readiness_analysis = recommendation_service.calculate_company_readiness(company_dict, user_problems)
    
    company = Company(
        id=company_id,
        user_id=current_user["id"],
        readiness_score=readiness_analysis["readiness_score"],
        skill_gaps=readiness_analysis["skill_gaps"],
        **company_data.model_dump()
    )
    
    doc = company.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    
    await db.companies.insert_one(doc)
    return company

@api_router.get("/companies", response_model=List[Company])
async def get_companies(current_user: dict = Depends(get_current_user)):
    companies = await db.companies.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    for c in companies:
        if isinstance(c.get("created_at"), str):
            c["created_at"] = datetime.fromisoformat(c["created_at"])
    return companies

@api_router.delete("/companies/{company_id}")
async def delete_company(company_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.companies.delete_one({"id": company_id, "user_id": current_user["id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return {"message": "Company deleted"}

@api_router.post("/applications", response_model=Application)
async def create_application(app_data: ApplicationCreate, current_user: dict = Depends(get_current_user)):
    app_id = str(uuid.uuid4())
    
    applied_date = app_data.applied_date or datetime.now(timezone.utc)
    
    application = Application(
        id=app_id,
        user_id=current_user["id"],
        applied_date=applied_date,
        **{k: v for k, v in app_data.model_dump().items() if k != "applied_date"}
    )
    
    doc = application.model_dump()
    doc["applied_date"] = doc["applied_date"].isoformat()
    doc["created_at"] = doc["created_at"].isoformat()
    if doc.get("interview_date"):
        doc["interview_date"] = doc["interview_date"].isoformat()
    
    await db.applications.insert_one(doc)
    return application

@api_router.get("/applications", response_model=List[Application])
async def get_applications(current_user: dict = Depends(get_current_user)):
    applications = await db.applications.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    for a in applications:
        if isinstance(a.get("applied_date"), str):
            a["applied_date"] = datetime.fromisoformat(a["applied_date"])
        if isinstance(a.get("created_at"), str):
            a["created_at"] = datetime.fromisoformat(a["created_at"])
        if a.get("interview_date") and isinstance(a["interview_date"], str):
            a["interview_date"] = datetime.fromisoformat(a["interview_date"])
    return applications

@api_router.patch("/applications/{app_id}", response_model=Application)
async def update_application(app_id: str, update_data: ApplicationUpdate, current_user: dict = Depends(get_current_user)):
    update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
    
    if "interview_date" in update_dict and update_dict["interview_date"]:
        update_dict["interview_date"] = update_dict["interview_date"].isoformat()
    
    if not update_dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")
    
    result = await db.applications.update_one(
        {"id": app_id, "user_id": current_user["id"]},
        {"$set": update_dict}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    
    app_doc = await db.applications.find_one({"id": app_id}, {"_id": 0})
    if isinstance(app_doc.get("applied_date"), str):
        app_doc["applied_date"] = datetime.fromisoformat(app_doc["applied_date"])
    if isinstance(app_doc.get("created_at"), str):
        app_doc["created_at"] = datetime.fromisoformat(app_doc["created_at"])
    if app_doc.get("interview_date") and isinstance(app_doc["interview_date"], str):
        app_doc["interview_date"] = datetime.fromisoformat(app_doc["interview_date"])
    
    return Application(**app_doc)

@api_router.delete("/applications/{app_id}")
async def delete_application(app_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.applications.delete_one({"id": app_id, "user_id": current_user["id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    return {"message": "Application deleted"}

@api_router.post("/mock-interviews", response_model=MockInterview)
async def create_mock_interview(interview_data: MockInterviewCreate, current_user: dict = Depends(get_current_user)):
    interview_id = str(uuid.uuid4())
    
    questions = await ai_service.generate_mock_interview_questions(
        interview_data.company_name,
        interview_data.role,
        interview_data.difficulty,
        count=5
    )
    
    interview = MockInterview(
        id=interview_id,
        user_id=current_user["id"],
        questions=questions,
        **interview_data.model_dump()
    )
    
    doc = interview.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    
    await db.mock_interviews.insert_one(doc)
    return interview

@api_router.get("/mock-interviews", response_model=List[MockInterview])
async def get_mock_interviews(current_user: dict = Depends(get_current_user)):
    interviews = await db.mock_interviews.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    for i in interviews:
        if isinstance(i.get("created_at"), str):
            i["created_at"] = datetime.fromisoformat(i["created_at"])
    return interviews

@api_router.post("/mock-interviews/{interview_id}/answer")
async def submit_interview_answer(
    interview_id: str,
    answer_data: MockInterviewAnswer,
    current_user: dict = Depends(get_current_user)
):
    interview = await db.mock_interviews.find_one({"id": interview_id, "user_id": current_user["id"]}, {"_id": 0})
    if not interview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    
    question = next((q for q in interview["questions"] if q["id"] == answer_data.question_id), None)
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    
    evaluation = await ai_service.evaluate_interview_answer(
        question["question"],
        answer_data.answer,
        question["difficulty"]
    )
    
    answer_record = {
        "question_id": answer_data.question_id,
        "answer": answer_data.answer,
        "evaluation": evaluation
    }
    
    await db.mock_interviews.update_one(
        {"id": interview_id},
        {"$push": {"answers": answer_record}}
    )
    
    return {"evaluation": evaluation, "message": "Answer submitted"}

@api_router.post("/mock-interviews/{interview_id}/complete")
async def complete_mock_interview(interview_id: str, current_user: dict = Depends(get_current_user)):
    interview = await db.mock_interviews.find_one({"id": interview_id, "user_id": current_user["id"]}, {"_id": 0})
    if not interview:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    
    answers = interview.get("answers", [])
    if not answers:
        overall_score = 0.0
    else:
        scores = [a["evaluation"]["score"] for a in answers]
        overall_score = sum(scores) / len(scores)
    
    feedback = f"Completed {len(answers)}/{len(interview['questions'])} questions with average score: {overall_score:.1f}%"
    
    await db.mock_interviews.update_one(
        {"id": interview_id},
        {"$set": {"completed": True, "overall_score": overall_score, "feedback": feedback}}
    )
    
    xp_gain = int(overall_score / 2)
    await db.users.update_one({"id": current_user["id"]}, {"$inc": {"xp": xp_gain}})
    
    return {"overall_score": overall_score, "feedback": feedback}

@api_router.post("/resume/analyze", response_model=ResumeAnalysis)
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: str = "Software Engineer",
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")
    
    pdf_bytes = await file.read()
    
    analysis_result = await resume_service.analyze_resume(pdf_bytes, target_role)
    
    analysis_id = str(uuid.uuid4())
    analysis = ResumeAnalysis(
        id=analysis_id,
        user_id=current_user["id"],
        filename=file.filename,
        extracted_text=analysis_result["extracted_text"],
        ats_score=analysis_result["ats_score"],
        keyword_gaps=analysis_result["keyword_gaps"],
        missing_skills=analysis_result["missing_skills"],
        suggestions=analysis_result["suggestions"]
    )
    
    doc = analysis.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    
    await db.resume_analyses.insert_one(doc)
    return analysis

@api_router.get("/resume/history", response_model=List[ResumeAnalysis])
async def get_resume_history(current_user: dict = Depends(get_current_user)):
    analyses = await db.resume_analyses.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    for a in analyses:
        if isinstance(a.get("created_at"), str):
            a["created_at"] = datetime.fromisoformat(a["created_at"])
    return analyses

@api_router.post("/ai-mentor/chat")
async def chat_with_mentor(chat_data: ChatRequest, current_user: dict = Depends(get_current_user)):
    session_id = chat_data.session_id or str(uuid.uuid4())
    
    user_context = {
        "xp": current_user.get("xp", 0),
        "level": current_user.get("level", 1),
        "streak": current_user.get("streak", 0),
        "total_problems": await db.problems.count_documents({"user_id": current_user["id"]})
    }
    
    response = await ai_service.ai_mentor_chat(chat_data.message, session_id, user_context)
    
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        user_id=current_user["id"],
        session_id=session_id,
        role="user",
        content=chat_data.message
    )
    
    ai_msg = ChatMessage(
        id=str(uuid.uuid4()),
        user_id=current_user["id"],
        session_id=session_id,
        role="assistant",
        content=response
    )
    
    user_doc = user_msg.model_dump()
    user_doc["created_at"] = user_doc["created_at"].isoformat()
    
    ai_doc = ai_msg.model_dump()
    ai_doc["created_at"] = ai_doc["created_at"].isoformat()
    
    await db.chat_messages.insert_many([user_doc, ai_doc])
    
    return {"response": response, "session_id": session_id}

@api_router.get("/ai-mentor/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str, current_user: dict = Depends(get_current_user)):
    messages = await db.chat_messages.find(
        {"user_id": current_user["id"], "session_id": session_id},
        {"_id": 0}
    ).sort("created_at", 1).to_list(1000)
    
    for m in messages:
        if isinstance(m.get("created_at"), str):
            m["created_at"] = datetime.fromisoformat(m["created_at"])
    
    return messages

app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    # NOTE: uvicorn's reload feature requires passing the app as an import string
    # (e.g. `python -m uvicorn server:app --reload`). When running via uvicorn.run(app,...)
    # keep reload disabled so `python server.py` works reliably.
    uvicorn.run(app, host="0.0.0.0", port=8000)