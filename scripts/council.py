#!/usr/bin/env python3
"""
ARENA-001: Multi-Model Council
Parallel model execution with voting/consensus.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Optional
from collections import Counter

class ModelCouncil:
    """Multi-model council for parallel execution and voting."""
    
    # All models use same LM Studio endpoint
    BASE_URL = "http://127.0.0.1:1234/v1/chat/completions"
    
    # Model configurations (EXACT IDs from LM Studio)
    MODELS = {
        'nerdsking-python-coder-3b-i': {'size': 2.2, 'weight': 1},
        'meta-llama-3.1-8b-instruct': {'size': 4.9, 'weight': 2},
        'strand-rust-coder-14b-v1': {'size': 9.0, 'weight': 3},
        'huihui-mistral-small-3.2-24b-instruct-2506-abliterated-v2': {'size': 15.3, 'weight': 4},
        'zai-org/glm-4.7-flash': {'size': 18.1, 'weight': 5},
        'qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive': {'size': 22.1, 'weight': 6},
    }
    
    def __init__(self, active_models: Optional[List[str]] = None, timeout: int = 60):
        """
        Initialize council.
        
        Args:
            active_models: List of model IDs to use. If None, uses small models only.
            timeout: Timeout for each model request (seconds)
        """
        # Default to small models for testing
        self.active_models = active_models or [
            'nerdsking-python-coder-3b-i',
            'meta-llama-3.1-8b-instruct'
        ]
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query_model(self, model: str, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Query single model."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            async with self.session.post(
                self.BASE_URL, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    print(f"⚠️  {model}: HTTP {response.status} - {error_text[:50]}")
                    return None
        except asyncio.TimeoutError:
            print(f"⏱️  {model}: Timeout")
            return None
        except Exception as e:
            print(f"❌ {model}: {e}")
            return None
    
    async def query_all(self, prompt: str, max_tokens: int = 200) -> Dict[str, str]:
        """Query all active models in parallel."""
        print(f"🤖 Querying {len(self.active_models)} models...")
        
        tasks = [
            self.query_model(model, prompt, max_tokens)
            for model in self.active_models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = {}
        for model, result in zip(self.active_models, results):
            if isinstance(result, str):
                responses[model] = result
                print(f"✅ {model}: {result[:50]}...")
            elif isinstance(result, Exception):
                print(f"❌ {model}: {result}")
        
        return responses
    
    def majority_vote(self, responses: Dict[str, str]) -> str:
        """Simple majority vote (exact match)."""
        if not responses:
            raise ValueError("No valid responses")
        
        # Count exact matches
        vote_counts = Counter(responses.values())
        winner, count = vote_counts.most_common(1)[0]
        
        print(f"\n📊 Majority Voting:")
        for resp, cnt in vote_counts.most_common():
            print(f"   {cnt} votes: {resp[:60]}...")
        
        print(f"\n🏆 Winner: {count}/{len(responses)} votes")
        return winner
    
    def weighted_vote(self, responses: Dict[str, str]) -> str:
        """Weighted vote by model size/quality."""
        if not responses:
            raise ValueError("No valid responses")
        
        # Calculate weighted scores
        scores = Counter()
        for model, response in responses.items():
            weight = self.MODELS.get(model, {}).get('weight', 1)
            scores[response] += weight
        
        winner, score = scores.most_common(1)[0]
        total_weight = sum(scores.values())
        
        print(f"\n📊 Weighted Voting:")
        for resp, sc in scores.most_common():
            pct = (sc / total_weight) * 100
            print(f"   {pct:.0f}%: {resp[:60]}...")
        
        print(f"\n🏆 Winner: {(score/total_weight)*100:.0f}% weight")
        return winner
    
    async def decide(self, prompt: str, strategy: str = "weighted", max_tokens: int = 200) -> str:
        """
        Get council decision.
        
        Args:
            prompt: Question for the council
            strategy: "majority" or "weighted"
            max_tokens: Max tokens per model response
        """
        responses = await self.query_all(prompt, max_tokens)
        
        if len(responses) == 0:
            return "Error: No models responded"
        elif len(responses) == 1:
            # Fallback to single response
            return list(responses.values())[0]
        
        if strategy == "majority":
            return self.majority_vote(responses)
        else:
            return self.weighted_vote(responses)


# Simple wrapper for non-async usage
def council_decide(prompt: str, models: Optional[List[str]] = None, strategy: str = "weighted") -> str:
    """Synchronous wrapper for council decision."""
    async def _decide():
        async with ModelCouncil(active_models=models) as council:
            return await council.decide(prompt, strategy)
    
    return asyncio.run(_decide())


if __name__ == "__main__":
    # Test with single model
    print("🧪 Testing ARENA Council\n")
    
    prompt = "What is Python? Answer in 5 words."
    result = council_decide(
        prompt,
        models=['nerdsking-python-coder-3b-i'],  # Single model test
        strategy="weighted"
    )
    print(f"\n📝 Result: {result}")
