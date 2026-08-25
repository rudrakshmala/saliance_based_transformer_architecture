import random
from typing import List, Dict

class MemoryUpdateBenchmarkGenerator:
    def __init__(self, num_samples: int = 1000, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        self.random = random.Random(seed)
        
        self.attributes = {
            'favorite_language': [
                ('Python', 'I primarily write in Python.'),
                ('Rust', 'I have switched to using Rust.'),
                ('Go', 'I now use Go for most of my projects.'),
                ('C++', 'I am doing C++ development mostly now.'),
                ('Java', 'My favorite language is Java.')
            ],
            'city': [
                ('New York', 'I live in New York.'),
                ('Austin', 'I recently moved to Austin.'),
                ('Seattle', 'I am currently based in Seattle.'),
                ('London', 'I have relocated to London.'),
                ('Tokyo', 'I am living in Tokyo now.')
            ],
            'job_title': [
                ('Engineer', 'I work as a Software Engineer.'),
                ('Senior Engineer', 'I just got promoted to Senior Engineer.'),
                ('Principal Engineer', 'I am now a Principal Engineer.'),
                ('Manager', 'I transitioned to an Engineering Manager role.'),
                ('Director', 'I am a Director of Engineering now.')
            ],
            'pet': [
                ('dog', 'I have a pet dog.'),
                ('cat', 'I adopted a cat recently.'),
                ('rabbit', 'I got a pet rabbit.'),
                ('parrot', 'I bought a parrot.'),
                ('hamster', 'I have a hamster now.')
            ],
            'diet': [
                ('omnivore', 'I eat everything.'),
                ('vegetarian', 'I became a vegetarian.'),
                ('vegan', 'I am strictly vegan now.'),
                ('pescatarian', 'I switched to a pescatarian diet.'),
                ('keto', 'I am on a keto diet these days.')
            ]
        }
        
        self.queries = {
            'favorite_language': 'What language do I primarily use?',
            'city': 'What city do I live in?',
            'job_title': 'What is my current job title?',
            'pet': 'What kind of pet do I have?',
            'diet': 'What is my diet?'
        }

    def generate(self) -> List[Dict]:
        samples = []
        for i in range(self.num_samples):
            attr = self.random.choice(list(self.attributes.keys()))
            num_updates = self.random.randint(2, 4)
            chosen_updates = self.random.sample(self.attributes[attr], num_updates)
            
            history = []
            stale_answers = []
            
            current_session = self.random.randint(1, 10)
            
            for j, (val, text) in enumerate(chosen_updates):
                history.append({
                    'session_idx': current_session,
                    'text': text,
                    'value': val,
                    'timestamp': current_session * 1000
                })
                if j < num_updates - 1:
                    stale_answers.append(val)
                current_session += self.random.randint(1, 5)
            
            correct_answer = chosen_updates[-1][0]
            query_session = current_session + self.random.randint(1, 5)
            
            samples.append({
                'sample_id': f'mem_update_{i:04d}',
                'attribute': attr,
                'update_history': history,
                'query_session': query_session,
                'query_text': self.queries[attr],
                'correct_answer': correct_answer,
                'stale_answers': stale_answers,
                'num_updates': num_updates,
                'is_overwrite': True
            })
            
        return samples
