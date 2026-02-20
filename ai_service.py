from typing import List, Dict, Any, Optional
import asyncio
import json
import os
import logging
import hashlib
import re

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.request_timeout_seconds = float(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "25"))
        self.max_input_chars = int(os.environ.get("AI_MAX_INPUT_CHARS", "12000"))

        self.enabled = bool(self.api_key)
        self.client: Optional[AsyncOpenAI] = None
        if self.enabled:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.cache: Dict[str, Any] = {}
        self.chat_histories: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_messages = int(os.environ.get("AI_MENTOR_MAX_HISTORY_MESSAGES", "20"))

    def _missing_key_message(self) -> str:
        return (
            "Groq is not configured. Set GROQ_API_KEY for the backend process (PowerShell: $env:GROQ_API_KEY=\"...\"). "
            "Then restart the backend."
        )

    def _local_fallback_response(self, user_message: str, user_context: Dict[str, Any]) -> str:
        if not self.enabled:
            return self._missing_key_message()

        msg = (user_message or "").strip()
        lower = msg.lower()

        xp = user_context.get("xp", 0)
        level = user_context.get("level", 1)
        total_problems = user_context.get("total_problems", 0)

        if any(k in lower for k in ["tip", "tips", "advice"]):
            return (
                "Here are 3 high-impact tips you can apply today:\n"
                "1) Pick 1 weak topic and solve 3 problems (easy→medium) back-to-back.\n"
                "2) After each problem, write a 5-line solution summary + 2 edge cases.\n"
                "3) Do 1 mock interview question and speak your approach out loud for 5 minutes."
            )

        if "study plan" in lower or "plan" in lower:
            return (
                "Quick 7-day plan (60–90 min/day):\n"
                "Day 1–2: Arrays/Strings + 6 problems\n"
                "Day 3: Hashing/Two pointers + 4 problems\n"
                "Day 4: Stack/Queue + 4 problems\n"
                "Day 5: Binary search + 4 problems\n"
                "Day 6: Trees basics + 3 problems\n"
                "Day 7: 1 mock interview + review notes"
            )

        if "resume" in lower:
            return (
                "Resume quick wins:\n"
                "- Rewrite bullets as: Action + Tech + Impact (numbers).\n"
                "- Put 4–6 strongest skills at the top (matching target roles).\n"
                "- Add 1 project with measurable outcome + link (GitHub/demo)."
            )

        return (
            "AI is temporarily unavailable (Groq error/timeout). "
            "Tell me your target role/company + weak topics and I’ll give a focused plan.\n\n"
            f"Current stats: XP={xp}, Level={level}, Problems solved={total_problems}."
        )

    async def _chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
        if not self.enabled or not self.client:
            raise RuntimeError("Groq is not configured. Set GROQ_API_KEY in backend/.env")

        try:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                ),
                timeout=self.request_timeout_seconds,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return content or ""
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"Groq chat.completions error: {e}")
            raise

    def _safe_json_loads(self, text: str):
        if not text:
            return None

        candidate = text.strip()

        # Common case: the model wraps JSON in markdown fences.
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()

        try:
            return json.loads(candidate)
        except Exception:
            pass

        # Try to extract the first JSON object/array substring.
        # This handles responses like: "Sure! Here is the JSON: { ... }" or code fences.
        obj_start = candidate.find("{")
        obj_end = candidate.rfind("}")
        arr_start = candidate.find("[")
        arr_end = candidate.rfind("]")

        substrings: List[str] = []
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            substrings.append(candidate[obj_start : obj_end + 1])
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            substrings.append(candidate[arr_start : arr_end + 1])

        for s in substrings:
            try:
                return json.loads(s)
            except Exception:
                continue

        return None

    def _stable_cache_key(self, prefix: str, *parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            if p is None:
                continue
            h.update(str(p).encode("utf-8", errors="ignore"))
            h.update(b"\0")
        digest = h.hexdigest()[:24]
        return f"{prefix}_{digest}"

    def _truncate_for_model(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        if len(t) <= self.max_input_chars:
            return t
        return t[: self.max_input_chars] + "\n\n[TRUNCATED: resume text was longer than AI_MAX_INPUT_CHARS]"

    async def ai_mentor_chat(self, user_message: str, session_id: str, user_context: Dict[str, Any]) -> str:
        if not self.enabled:
            return self._missing_key_message()

        system_prompt = f"""You are an expert AI Placement Strategist and Career Mentor for students preparing for technical interviews.

Your role is to:
- Analyze their preparation progress and identify weak areas
- Provide strategic guidance on interview preparation
- Answer DSA and technical interview questions
- Suggest personalized study plans
- Predict readiness for specific companies
- Offer career and resume advice

User Context:
- XP: {user_context.get('xp', 0)}
- Level: {user_context.get('level', 1)}
- Streak: {user_context.get('streak', 0)}
- Total Problems Solved: {user_context.get('total_problems', 0)}

Be concise, actionable, and strategic. Focus on high-impact advice.
"""

        history = self.chat_histories.get(session_id, [])
        history = history[-self.max_history_messages :]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            reply = await self._chat_completion(messages, temperature=0.6)
            new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
            self.chat_histories[session_id] = new_history[-self.max_history_messages :]
            return reply
        except Exception:
            return self._local_fallback_response(user_message, user_context)

    async def analyze_resume(self, resume_text: str, target_role: str = "Software Engineer") -> Dict[str, Any]:
        if not self.enabled:
            return {
                "ats_score": 0.0,
                "keyword_gaps": ["Groq not configured"],
                "missing_skills": [],
                "suggestions": [self._missing_key_message()],
            }

        resume_text = self._truncate_for_model(resume_text)
        cache_key = self._stable_cache_key("resume", resume_text, target_role, self.model)
        if cache_key in self.cache:
            return self.cache[cache_key]

        messages = [
            {"role": "system", "content": "You are an expert ATS and Resume Analyzer for technical roles."},
            {
                "role": "user",
                "content": f"""Analyze this resume for a {target_role} position:

{resume_text}

Return ONLY valid JSON with this shape:
{{
  "ats_score": 0-100,
  "keyword_gaps": ["..."],
  "missing_skills": ["..."],
  "suggestions": ["..."]
}}
""",
            },
        ]

        try:
            text = await self._chat_completion(messages, temperature=0.2)
            parsed = self._safe_json_loads(text)
            if isinstance(parsed, dict):
                # Ensure required keys exist; be tolerant of partial output.
                parsed.setdefault("ats_score", 0)
                parsed.setdefault("keyword_gaps", [])
                parsed.setdefault("missing_skills", [])
                parsed.setdefault("suggestions", [])
            else:
                raise ValueError("Resume analysis was not valid JSON object")
            self.cache[cache_key] = parsed
            return parsed
        except Exception as e:
            logger.warning(f"Resume analysis fallback: {e}")
            return {
                "ats_score": 70.0,
                "keyword_gaps": ["AI response not available/parsable"],
                "missing_skills": [],
                "suggestions": ["Try again after fixing GROQ_API_KEY/quota."],
            }

    async def generate_mock_interview_questions(self, company: str, role: str, difficulty: str, count: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            return [
                {
                    "id": i,
                    "question": self._missing_key_message(),
                    "type": "coding",
                    "difficulty": difficulty,
                    "topics": ["configuration"],
                }
                for i in range(1, count + 1)
            ]

        cache_key = self._stable_cache_key("mock", company, role, difficulty, str(count), self.model)
        if cache_key in self.cache:
            return self.cache[cache_key]

        messages = [
            {"role": "system", "content": "You are an expert technical interviewer."},
            {
                "role": "user",
                "content": f"""Generate {count} technical interview questions for:
Company: {company}
Role: {role}
Difficulty: {difficulty}

Return ONLY valid JSON array with objects:
[
  {{
    "id": 1,
    "question": "...",
    "type": "coding/system_design/behavioral",
    "difficulty": "{difficulty}",
    "topics": ["..."]
  }}
]
""",
            },
        ]

        try:
            text = await self._chat_completion(messages, temperature=0.5)
            parsed = self._safe_json_loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                parsed = parsed["questions"]
            if not isinstance(parsed, list):
                raise ValueError("Questions were not a JSON list")

            # Normalize question IDs to ints where possible.
            normalized: List[Dict[str, Any]] = []
            for i, q in enumerate(parsed, start=1):
                if not isinstance(q, dict):
                    continue
                q.setdefault("id", i)
                try:
                    q["id"] = int(q["id"])
                except Exception:
                    q["id"] = i
                q.setdefault("difficulty", difficulty)
                q.setdefault("type", "coding")
                q.setdefault("topics", [])
                normalized.append(q)

            if not normalized:
                raise ValueError("Questions list was empty after normalization")

            parsed = normalized
            self.cache[cache_key] = parsed
            return parsed
        except Exception as e:
            logger.warning(f"Mock questions fallback: {e}")
            return [
                {
                    "id": i,
                    "question": f"Sample {difficulty} question {i} for {company}",
                    "type": "coding",
                    "difficulty": difficulty,
                    "topics": ["algorithms"],
                }
                for i in range(1, count + 1)
            ]

    async def evaluate_interview_answer(self, question: str, answer: str, difficulty: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "score": 0.0,
                "strengths": [],
                "weaknesses": ["Groq not configured"],
                "suggestions": [self._missing_key_message()],
                "depth_rating": 1,
                "clarity_rating": 1,
                "correctness_rating": 1,
            }

        messages = [
            {"role": "system", "content": "You are an expert technical interviewer evaluating candidate responses."},
            {
                "role": "user",
                "content": f"""Question: {question}

Candidate's Answer: {answer}

Difficulty Level: {difficulty}

Return ONLY valid JSON:
{{
  "score": 0-100,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "depth_rating": 1-5,
  "clarity_rating": 1-5,
  "correctness_rating": 1-5
}}
""",
            },
        ]

        try:
            text = await self._chat_completion(messages, temperature=0.3)
            parsed = self._safe_json_loads(text)
            if isinstance(parsed, dict):
                parsed.setdefault("score", 0)
                parsed.setdefault("strengths", [])
                parsed.setdefault("weaknesses", [])
                parsed.setdefault("suggestions", [])
                parsed.setdefault("depth_rating", 1)
                parsed.setdefault("clarity_rating", 1)
                parsed.setdefault("correctness_rating", 1)
            else:
                raise ValueError("Evaluation was not valid JSON object")
            return parsed
        except Exception as e:
            logger.warning(f"Evaluation fallback: {e}")
            return {
                "score": 70.0,
                "strengths": ["Attempted the question"],
                "weaknesses": ["AI response not available/parsable"],
                "suggestions": ["Try again after fixing GROQ_API_KEY/quota."],
                "depth_rating": 3,
                "clarity_rating": 3,
                "correctness_rating": 3,
            }

    async def generate_study_plan(self, user_data: Dict[str, Any], weak_topics: List[str]) -> str:
        if not self.enabled:
            return self._missing_key_message()

        messages = [
            {"role": "system", "content": "You are an expert placement preparation coach."},
            {
                "role": "user",
                "content": f"""Create a personalized 7-day study plan.

User:
- Level: {user_data.get('level', 1)}
- XP: {user_data.get('xp', 0)}
- Problems Solved: {user_data.get('total_problems', 0)}

Weak Topics: {", ".join(weak_topics)}

Return a concise day-by-day plan with specific goals.""",
            },
        ]
        try:
            return await self._chat_completion(messages, temperature=0.6)
        except Exception:
            return "Day 1–7 plan unavailable right now. Fix GROQ_API_KEY/model and retry."


ai_service = AIService()