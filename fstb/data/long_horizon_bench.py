import random
from typing import List, Dict, Tuple

class LongHorizonBenchmarkGenerator:
    def __init__(self, num_samples: int = 500, chain_length_range: Tuple[int, int] = (50, 200), seed: int = 42):
        self.num_samples = num_samples
        self.chain_length_range = chain_length_range
        self.seed = seed
        self.random = random.Random(seed)
        self.chain_types = ['project_dependency', 'preference_evolution', 'commitment_tracking']

    def generate(self) -> List[Dict]:
        samples = []
        for i in range(self.num_samples):
            chain_type = self.random.choice(self.chain_types)
            chain_length = self.random.randint(self.chain_length_range[0], self.chain_length_range[1])
            
            if chain_type == 'project_dependency':
                sample = self._gen_project_dependency(i, chain_length)
            elif chain_type == 'preference_evolution':
                sample = self._gen_preference_evolution(i, chain_length)
            else:
                sample = self._gen_commitment_tracking(i, chain_length)
                
            samples.append(sample)
            
        return samples

    def _gen_project_dependency(self, idx: int, length: int) -> Dict:
        start_session = self.random.randint(1, 20)
        dep_session = start_session + self.random.randint(10, 30)
        cancel_session = dep_session + self.random.randint(20, length - dep_session - 10)
        query_session = length
        
        events = [
            {'session_idx': start_session, 'text': 'I started Project Alpha.', 'event_type': 'introduction', 'entity': 'Project Alpha'},
            {'session_idx': dep_session, 'text': 'Project Alpha depends on Library X.', 'event_type': 'dependency', 'entity': 'Library X'},
            {'session_idx': cancel_session, 'text': 'Library X was deprecated.', 'event_type': 'cancellation', 'entity': 'Library X'}
        ]
        
        return {
            'sample_id': f'long_horiz_proj_{idx:04d}',
            'chain_type': 'project_dependency',
            'chain_length': length,
            'events': events,
            'query_session': query_session,
            'query': 'Can I still use Library X for Project Alpha?',
            'correct_answer': f'No, Library X was deprecated in session {cancel_session}',
            'reasoning_chain': [
                'Project Alpha started',
                'Project Alpha added dependency on Library X',
                'Library X was deprecated'
            ],
            'required_sessions': [start_session, dep_session, cancel_session]
        }

    def _gen_preference_evolution(self, idx: int, length: int) -> Dict:
        s1 = self.random.randint(1, length//3)
        s2 = self.random.randint(length//3 + 1, (2*length)//3)
        query_session = length
        
        events = [
            {'session_idx': s1, 'text': 'I prefer dark mode.', 'event_type': 'introduction', 'entity': 'theme'},
            {'session_idx': s2, 'text': 'I now prefer light mode for better readability.', 'event_type': 'update', 'entity': 'theme'}
        ]
        
        return {
            'sample_id': f'long_horiz_pref_{idx:04d}',
            'chain_type': 'preference_evolution',
            'chain_length': length,
            'events': events,
            'query_session': query_session,
            'query': 'What theme do I prefer?',
            'correct_answer': 'Light mode',
            'reasoning_chain': ['Initial preference was dark mode', 'Updated preference to light mode'],
            'required_sessions': [s1, s2]
        }
        
    def _gen_commitment_tracking(self, idx: int, length: int) -> Dict:
        s1 = self.random.randint(1, length//3)
        s2 = self.random.randint(length//3 + 1, (2*length)//3)
        query_session = length
        
        events = [
            {'session_idx': s1, 'text': 'I promised to finish the report by Friday.', 'event_type': 'commitment', 'entity': 'report'},
            {'session_idx': s2, 'text': 'I sent you the finished report.', 'event_type': 'completion', 'entity': 'report'}
        ]
        
        return {
            'sample_id': f'long_horiz_commit_{idx:04d}',
            'chain_type': 'commitment_tracking',
            'chain_length': length,
            'events': events,
            'query_session': query_session,
            'query': 'Is the report finished?',
            'correct_answer': 'Yes, it was sent in a previous session.',
            'reasoning_chain': ['Committed to report', 'Report was finished and sent'],
            'required_sessions': [s1, s2]
        }
