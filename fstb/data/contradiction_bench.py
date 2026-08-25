import random
from typing import List, Dict

class ContradictionBenchmarkGenerator:
    def __init__(self, num_samples: int = 1000, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self.random = random.Random(seed)
        
        self.categories = {
            'company': [
                ('I work at Google.', 'I just joined Meta.', 'What company do you work for?'),
                ('I am employed by Amazon.', 'I started working at Microsoft.', 'Where are you currently employed?'),
            ],
            'city': [
                ('I live in London.', 'I moved to Tokyo.', 'What city do you live in?'),
                ('My home is in Paris.', 'I relocated to Berlin.', 'Where is your home?'),
            ],
            'relationship': [
                ('I am single.', 'I got married.', 'What is your relationship status?'),
                ('I have a boyfriend.', 'We broke up, I am single now.', 'Are you currently dating anyone?'),
            ],
            'dietary': [
                ('I am vegan.', 'I started eating meat again.', 'What is your diet?'),
                ('I am lactose intolerant.', 'I am fine with dairy now.', 'Can you eat dairy?'),
            ],
            'project_status': [
                ('Project X is ongoing.', 'Project X was cancelled.', 'What is the status of Project X?'),
                ('I am working on the Alpha initiative.', 'The Alpha initiative is complete.', 'Is the Alpha initiative finished?'),
            ]
        }
        
        self.severities = ['soft', 'hard', 'temporal']

    def generate(self) -> List[Dict]:
        samples = []
        for i in range(self.num_samples):
            category = self.random.choice(list(self.categories.keys()))
            initial, contradict, query = self.random.choice(self.categories[category])
            
            session_gap = self.random.randint(1, 20)
            severity = self.random.choice(self.severities)
            
            samples.append({
                'sample_id': f'contra_{i:04d}',
                'category': category,
                'initial_statement': initial,
                'contradicting_statement': contradict,
                'session_gap': session_gap,
                'query': query,
                'correct_answer': contradict, 
                'stale_answer': initial,
                'is_contradiction': True,
                'contradiction_severity': severity
            })
            
        return samples
