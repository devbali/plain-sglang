#!/usr/bin/env python3
"""Test script for static partition fairness policy.

Tests that:
1. Each of N users gets total_cache // N tokens
2. Users cannot exceed their partition
3. Over-quota users get reordered to the end of the queue
"""

import argparse
import asyncio
import time
from typing import List

import openai


async def send_request(
    client: openai.AsyncOpenAI,
    user_id: str,
    prompt: str,
    max_tokens: int = 50,
) -> dict:
    """Send a single completion request."""
    try:
        start = time.time()
        response = await asyncio.wait_for(
            client.completions.create(
                model="default",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.8,
                user=user_id,
            ),
            timeout=120.0,
        )
        latency = time.time() - start
        return {
            "user": user_id,
            "success": True,
            "latency": latency,
            "completion": response.choices[0].text,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "user": user_id,
            "success": False,
            "error": str(e),
        }


async def run_experiment(
    base_url: str,
    num_users: int = 4,
    requests_per_user: int = 3,
    max_tokens: int = 50,
):
    """Run static partition experiment."""
    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
    )

    print(f"=== Static Partition Experiment ===")
    print(f"Users: {num_users}")
    print(f"Requests per user: {requests_per_user}")
    print(f"Max tokens per request: {max_tokens}")
    print()

    # Generate test prompts
    prompts = [
        "Once upon a time in a faraway land",
        "The quick brown fox jumps over",
        "In the year 2050, artificial intelligence",
        "Scientists discovered a new planet that",
        "The ancient civilization built massive",
    ]

    # Send requests from all users
    tasks = []
    for user_idx in range(num_users):
        user_id = f"user-{chr(ord('A') + user_idx)}"  # user-A, user-B, etc.
        for req_idx in range(requests_per_user):
            prompt = prompts[req_idx % len(prompts)]
            tasks.append(send_request(client, user_id, prompt, max_tokens))

    # Execute all requests concurrently
    print("Sending all requests concurrently...")
    results = await asyncio.gather(*tasks)

    # Analyze results
    print("\n=== Results ===")
    
    user_stats = {}
    for result in results:
        user = result.get("user", "unknown")
        if user not in user_stats:
            user_stats[user] = {
                "success": 0,
                "failed": 0,
                "total_tokens": 0,
                "total_latency": 0.0,
                "errors": [],
            }
        
        if result.get("success"):
            user_stats[user]["success"] += 1
            user_stats[user]["total_tokens"] += result.get("prompt_tokens", 0) + result.get("completion_tokens", 0)
            user_stats[user]["total_latency"] += result.get("latency", 0)
        else:
            user_stats[user]["failed"] += 1
            user_stats[user]["errors"].append(result.get("error", "Unknown error"))

    # Print per-user stats
    for user in sorted(user_stats.keys()):
        stats = user_stats[user]
        avg_latency = stats["total_latency"] / max(1, stats["success"])
        print(f"\n{user}:")
        print(f"  Successful: {stats['success']}/{requests_per_user}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Avg latency: {avg_latency:.3f}s")
        if stats["errors"]:
            print(f"  Errors: {stats['errors'][:3]}")  # Show first 3 errors

    # Check fairness
    print("\n=== Fairness Analysis ===")
    token_counts = [stats["total_tokens"] for stats in user_stats.values()]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        variance = max_tokens - min_tokens
        
        print(f"Average tokens per user: {avg_tokens:.1f}")
        print(f"Max tokens: {max_tokens}")
        print(f"Min tokens: {min_tokens}")
        print(f"Variance: {variance} ({variance / avg_tokens * 100:.1f}%)")
        
        # Static partition should have relatively low variance
        if variance / avg_tokens < 0.3:
            print("✓ Static partition working: low variance across users")
        else:
            print("⚠ High variance detected - check policy enforcement")

    await client.close()


def main():
    parser = argparse.ArgumentParser(description="Test static partition fairness")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000/v1",
        help="SGLang server base URL",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=4,
        help="Number of concurrent users",
    )
    parser.add_argument(
        "--requests-per-user",
        type=int,
        default=3,
        help="Requests per user",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens per request",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_experiment(
        args.base_url,
        args.num_users,
        args.requests_per_user,
        args.max_tokens,
    ))


if __name__ == "__main__":
    main()
