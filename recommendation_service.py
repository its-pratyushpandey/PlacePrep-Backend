from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta
import math

class RecommendationService:
    @staticmethod
    def calculate_readiness_score(user_data: Dict[str, Any], problems: List[Dict], subjects: List[Dict]) -> float:
        if not problems:
            return 0.0
        
        total_problems = len(problems)
        difficulty_weights = {"easy": 1, "medium": 2, "hard": 3}
        
        weighted_score = sum(difficulty_weights.get(p.get("difficulty", "easy"), 1) for p in problems)
        max_possible = total_problems * 3
        
        base_score = (weighted_score / max_possible) * 100 if max_possible > 0 else 0
        
        streak_bonus = min(user_data.get("streak", 0) * 2, 20)
        
        readiness = min(base_score + streak_bonus, 100)
        return round(readiness, 2)
    
    @staticmethod
    def detect_weak_topics(problems: List[Dict]) -> List[str]:
        if not problems:
            return ["Arrays", "Dynamic Programming", "Graphs"]
        
        topic_counts = {}
        for problem in problems:
            for topic in problem.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if not topic_counts:
            return ["No topics tracked yet"]
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1])
        weak_topics = [topic for topic, count in sorted_topics[:3]]
        
        return weak_topics if weak_topics else ["Keep solving more problems"]
    
    @staticmethod
    def detect_strong_topics(problems: List[Dict]) -> List[str]:
        if not problems:
            return []
        
        topic_counts = {}
        for problem in problems:
            for topic in problem.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if not topic_counts:
            return []
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        strong_topics = [topic for topic, count in sorted_topics[:3] if count >= 3]
        
        return strong_topics
    
    @staticmethod
    def calculate_interview_probability(readiness_score: float, applications: List[Dict]) -> float:
        base_probability = readiness_score * 0.6
        
        if applications:
            interview_apps = len([a for a in applications if a.get("status") in ["screening", "interview", "offer"]])
            total_apps = len(applications)
            conversion_rate = (interview_apps / total_apps) * 40 if total_apps > 0 else 0
            base_probability += conversion_rate
        
        return min(round(base_probability, 2), 100)
    
    @staticmethod
    def calculate_application_conversion_rate(applications: List[Dict]) -> float:
        if not applications:
            return 0.0
        
        success_statuses = ["interview", "offer"]
        successful = len([a for a in applications if a.get("status") in success_statuses])
        total = len(applications)
        
        return round((successful / total) * 100, 2) if total > 0 else 0.0
    
    @staticmethod
    def calculate_skill_velocity(problems: List[Dict]) -> float:
        if len(problems) < 2:
            return 0.0
        
        recent_problems = [p for p in problems if "solved_at" in p]
        recent_problems.sort(key=lambda x: x["solved_at"], reverse=True)
        
        if len(recent_problems) < 2:
            return 0.0
        
        last_7_days = [p for p in recent_problems if isinstance(p["solved_at"], datetime) and 
                       (datetime.now(timezone.utc) - p["solved_at"]).days <= 7]
        
        velocity = len(last_7_days) * 10
        return min(round(velocity, 2), 100)
    
    @staticmethod
    def generate_next_action(readiness_score: float, weak_topics: List[str], applications: List[Dict]) -> str:
        if readiness_score < 30:
            return f"Focus on strengthening {weak_topics[0] if weak_topics else 'fundamentals'}. Solve 5 problems today."
        elif readiness_score < 60:
            return f"You're making progress! Practice {weak_topics[0] if weak_topics else 'medium'} difficulty problems."
        elif readiness_score < 80:
            return "Great momentum! Start applying to companies and schedule mock interviews."
        else:
            if len(applications) < 5:
                return "You're interview-ready! Apply to 5 more companies this week."
            else:
                return "Excellent preparation! Focus on interview practice and system design."
    
    @staticmethod
    def calculate_company_readiness(company: Dict[str, Any], user_problems: List[Dict]) -> Dict[str, Any]:
        required_skills = set(company.get("required_skills", []))
        
        user_topics = set()
        for problem in user_problems:
            user_topics.update(problem.get("topics", []))
        
        if not required_skills:
            readiness = 50.0
            skill_gaps = []
        else:
            matched_skills = required_skills.intersection(user_topics)
            readiness = (len(matched_skills) / len(required_skills)) * 100
            skill_gaps = list(required_skills - user_topics)
        
        return {
            "readiness_score": round(readiness, 2),
            "skill_gaps": skill_gaps,
            "matched_skills": list(matched_skills) if required_skills else []
        }

recommendation_service = RecommendationService()