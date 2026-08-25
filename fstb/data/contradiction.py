import random
from typing import List, Dict, Any

class ContradictionDatasetGenerator:
    """
    Generates datasets testing contradiction resolution and fact updating.
    Pairs initial statements with contradictory updates and checks if model suppresses stale facts.
    """
    COMPANIES = ["Acme Corp", "TechNova", "Starlight Inc", "Cyberdyne", "Global Dynamics"]
    PETS = ["dog named Max", "cat named Whiskers", "parrot named Echo", "golden retriever named Buddy"]
    
    def __init__(self, num_samples: int = 100, seed: int = 42):
        random.seed(seed)
        self.num_samples = num_samples

    def generate(self) -> List[Dict[str, Any]]:
        samples = []
        for i in range(self.num_samples):
            c1, c2 = random.sample(self.COMPANIES, 2)
            p1, p2 = random.sample(self.PETS, 2)

            # Type 1: Job Contradiction
            sample_job = {
                "id": f"job_contra_{i}",
                "fact_type": "job",
                "initial_fact": f"I work at {c1}.",
                "contradictory_fact": f"I left {c1} and joined {c2} last month.",
                "query": "Where do I currently work?",
                "target_answer": c2,
                "stale_answer": c1,
                "is_contradiction": True
            }

            # Type 2: Pet Contradiction
            sample_pet = {
                "id": f"pet_contra_{i}",
                "fact_type": "pet",
                "initial_fact": f"I have a {p1}.",
                "contradictory_fact": f"I had to rehome my pet and now I adopted a {p2}.",
                "query": "What pet do I have now?",
                "target_answer": p2,
                "stale_answer": p1,
                "is_contradiction": True
            }

            samples.extend([sample_job, sample_pet])
        return samples
