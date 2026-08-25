import random
from typing import List, Dict, Any

class LongTermConversationGenerator:
    """
    Simulates multi-session user conversations across 50 to 500 sessions.
    Tracks user traits: name, job, project, goals, relationships, and evolving facts over time.
    """
    NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley"]
    JOBS = ["Software Engineer", "Data Scientist", "Product Manager", "UX Designer", "Research Scientist"]
    PROJECTS = ["Project Alpha", "Project Quantum", "Thesis Experiment", "LLM Framework", "Cybersecurity Audit"]
    CITIES = ["New York", "San Francisco", "London", "Tokyo", "Berlin", "Delhi", "Mumbai"]

    def __init__(self, num_users: int = 5, num_sessions_per_user: int = 50, seed: int = 42):
        random.seed(seed)
        self.num_users = num_users
        self.num_sessions_per_user = num_sessions_per_user

    def generate(self) -> List[Dict[str, Any]]:
        dataset = []
        for user_idx in range(self.num_users):
            user_name = random.choice(self.NAMES)
            job = random.choice(self.JOBS)
            project = random.choice(self.PROJECTS)
            city = random.choice(self.CITIES)
            
            persona = {
                "user_id": f"user_{user_idx}",
                "name": user_name,
                "job": job,
                "project": project,
                "city": city
            }

            sessions = []
            for s in range(self.num_sessions_per_user):
                if s == 0:
                    text = f"Hello! My name is {user_name}. I work as a {job} and live in {city}."
                    fact_type = "persona_init"
                    importance = 1.0
                elif s == 10:
                    project_update = f"I am currently working on {project}."
                    text = f"Session {s}: {project_update}"
                    fact_type = "project_init"
                    importance = 0.9
                elif s == 25:
                    new_city = random.choice([c for c in self.CITIES if c != city])
                    city = new_city
                    text = f"Session {s}: I moved to {city} recently."
                    fact_type = "city_update"
                    importance = 0.95
                else:
                    text = f"Session {s}: Asking about {project} updates and standard queries."
                    fact_type = "general"
                    importance = 0.2

                sessions.append({
                    "session_id": s,
                    "text": text,
                    "fact_type": fact_type,
                    "importance": importance,
                    "current_persona_state": dict(persona)
                })

            dataset.append({
                "user_id": persona["user_id"],
                "sessions": sessions
            })
        return dataset
