import random
from typing import List, Dict, Any

class TemporalReasoningGenerator:
    """
    Generates temporal reasoning sequences.
    Example:
    t1: "Yesterday I was in Delhi."
    t2: "Today I am in Mumbai."
    query: "Where am I now?" -> Target: Mumbai
    """
    CITIES = ["Delhi", "Mumbai", "Bangalore", "London", "New York", "Paris", "Tokyo"]

    def __init__(self, num_samples: int = 100, seed: int = 42):
        random.seed(seed)
        self.num_samples = num_samples

    def generate(self) -> List[Dict[str, Any]]:
        samples = []
        for i in range(self.num_samples):
            city_yest, city_today = random.sample(self.CITIES, 2)
            samples.append({
                "id": f"temp_{i}",
                "context_t1": f"Yesterday I was in {city_yest}.",
                "context_t2": f"Today I arrived in {city_today}.",
                "query": "Where am I currently located?",
                "target_answer": city_today,
                "previous_answer": city_yest,
                "timestamp_t1": 1.0,
                "timestamp_t2": 2.0
            })
        return samples

class PreferenceEvolutionGenerator:
    """
    Generates preference shift datasets.
    Example:
    "I dislike coffee." -> "I started enjoying coffee recently."
    Query: "Do I like coffee?" -> Target: Yes / Enjoy coffee
    """
    ITEMS = [
        ("coffee", "disliked", "started enjoying"),
        ("spicy food", "hated", "love now"),
        ("waking up early", "struggled with", "now enjoy"),
        ("dark mode", "avoided", "use everywhere now")
    ]

    def __init__(self, num_samples: int = 100, seed: int = 42):
        random.seed(seed)
        self.num_samples = num_samples

    def generate(self) -> List[Dict[str, Any]]:
        samples = []
        for i in range(self.num_samples):
            item, old_pref, new_pref = random.choice(self.ITEMS)
            samples.append({
                "id": f"pref_{i}",
                "item": item,
                "past_statement": f"I used to {old_pref} {item}.",
                "recent_statement": f"Actually, I {new_pref} {item} recently.",
                "query": f"What is my current preference regarding {item}?",
                "target_preference": f"Likes/Enjoys {item}",
                "stale_preference": f"Dislikes {item}"
            })
        return samples

class MultiSessionProjectGenerator:
    """
    Simulates a software project developed across multiple sessions.
    Tracks architectural decisions, APIs, file structures, and commitments.
    """
    FRAMEWORKS = ["FastAPI", "Express", "Django", "Spring Boot", "Next.js"]
    DATABASES = ["PostgreSQL", "MongoDB", "Redis", "Cassandra", "SQLite"]

    def __init__(self, num_projects: int = 20, num_sessions: int = 10, seed: int = 42):
        random.seed(seed)
        self.num_projects = num_projects
        self.num_sessions = num_sessions

    def generate(self) -> List[Dict[str, Any]]:
        projects = []
        for p in range(self.num_projects):
            fw = random.choice(self.FRAMEWORKS)
            db = random.choice(self.DATABASES)
            proj_name = f"Project_{p}"
            
            sessions = [
                {"session": 1, "text": f"We decided to build {proj_name} using {fw} and {db}."},
                {"session": 3, "text": f"The primary API endpoint is /api/v1/data returning JSON."},
                {"session": 5, "text": f"We refactored database connection pooling to use {db} async driver."},
                {"session": 8, "text": f"Architecture rule: all authentication must go through JWT middleware."}
            ]
            projects.append({
                "project_id": proj_name,
                "framework": fw,
                "database": db,
                "sessions": sessions,
                "test_queries": [
                    {"query": f"What framework is used in {proj_name}?", "target": fw},
                    {"query": f"What database is used in {proj_name}?", "target": db},
                    {"query": "What is the authentication requirement?", "target": "JWT middleware"}
                ]
            })
        return projects
