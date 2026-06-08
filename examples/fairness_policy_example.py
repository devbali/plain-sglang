"""Example: Fairness policies using scheduling hooks.

This example demonstrates how to implement fairness policies using:
1. on_schedule_prefill: Control which requests get prefilled
2. on_prefill_vs_decode_decision: Control whether to run prefill or decode

Available example policies:
- SimpleFairnessPolicy: Deprioritize users already running
- ThrottlingPolicy: Per-user concurrency limits
- HybridPolicy: Fairness + throttling combined
- TimeSlicingPolicy: Alternate prefill/decode based on time windows
- StarvationPreventionPolicy: Force decode when running requests starved

Usage:
    from examples.fairness_policy_example import SimpleFairnessPolicy
    
    scheduler.scheduling_hooks_policy = SimpleFairnessPolicy()
    
For advanced examples using scheduler helpers (get_num_allocatable_reqs, etc.),
see SCHEDULING_HOOK_DESIGN.md.
"""

from typing import TYPE_CHECKING, List
from sglang.srt.scheduling_hooks import NoOpSchedulingPolicy

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SimpleFairnessPolicy(NoOpSchedulingPolicy):
    """Simple fairness policy: deprioritize users already running."""

    def __init__(self):
        super().__init__()
        # Track total tokens served per user
        self.user_tokens_served = {}
        
    def on_new_request(self, req):
        """Initialize tracking for new users."""
        if req.uid and req.uid not in self.user_tokens_served:
            self.user_tokens_served[req.uid] = 0
    
    def on_schedule_prefill(self, waiting_queue, running_batch, prefill_adder):
        """Reorder queue to prioritize users without running requests.
        
        Priority groups (lower number = higher priority):
        1. Users with no running requests, sorted by tokens served (ascending)
        2. Users already running, sorted by tokens served (ascending)
        """
        # Count running requests per user
        running_users = set()
        for req in running_batch.reqs:
            if req.uid:
                running_users.add(req.uid)
        
        # Define priority key
        def priority_key(req):
            uid = req.uid or "anonymous"
            tokens_served = self.user_tokens_served.get(uid, 0)
            
            # Group 0: users not running (prioritize)
            # Group 1: users already running (defer)
            group = 1 if uid in running_users else 0
            
            return (group, tokens_served)
        
        # Return reordered queue
        return sorted(waiting_queue, key=priority_key)
    
    def on_prefill_decision(self, batch):
        """Track tokens allocated to each user in this batch."""
        for req in batch.reqs:
            if req.uid:
                # Estimate tokens: input + expected output
                input_tokens = len(req.origin_input_ids) + len(req.output_ids)
                expected_output = req.sampling_params.max_new_tokens
                
                self.user_tokens_served[req.uid] = (
                    self.user_tokens_served.get(req.uid, 0) 
                    + input_tokens 
                    + expected_output
                )
    
    def on_decode_decision(self, batch):
        """Track decode tokens served (one token per request per step)."""
        for req in batch.reqs:
            if req.uid:
                self.user_tokens_served[req.uid] = (
                    self.user_tokens_served.get(req.uid, 0) + 1
                )


class ThrottlingPolicy(NoOpSchedulingPolicy):
    """Per-user throttling: max concurrent requests per user."""
    
    def __init__(self, max_concurrent_per_user=2):
        super().__init__()
        self.max_concurrent = max_concurrent_per_user
    
    def on_schedule_prefill(self, waiting_queue, running_batch, prefill_adder):
        """Filter out requests from users already at their concurrency limit."""
        # Count running requests per user
        running_counts = {}
        for req in running_batch.reqs:
            if req.uid:
                running_counts[req.uid] = running_counts.get(req.uid, 0) + 1
        
        # Filter queue
        filtered = []
        for req in waiting_queue:
            uid = req.uid or "anonymous"
            if running_counts.get(uid, 0) < self.max_concurrent:
                filtered.append(req)
        
        return filtered


class HybridPolicy(NoOpSchedulingPolicy):
    """Hybrid: fairness + throttling."""
    
    def __init__(self, max_concurrent_per_user=2):
        super().__init__()
        self.max_concurrent = max_concurrent_per_user
        self.user_tokens_served = {}
    
    def on_new_request(self, req):
        if req.uid and req.uid not in self.user_tokens_served:
            self.user_tokens_served[req.uid] = 0
    
    def on_schedule_prefill(self, waiting_queue, running_batch, prefill_adder):
        """Apply throttling, then fairness reordering."""
        # Step 1: Throttle
        running_counts = {}
        for req in running_batch.reqs:
            if req.uid:
                running_counts[req.uid] = running_counts.get(req.uid, 0) + 1
        
        filtered = [
            req for req in waiting_queue
            if running_counts.get(req.uid or "anonymous", 0) < self.max_concurrent
        ]
        
        # Step 2: Reorder by fairness
        running_users = set(req.uid for req in running_batch.reqs if req.uid)
        
        def priority_key(req):
            uid = req.uid or "anonymous"
            tokens_served = self.user_tokens_served.get(uid, 0)
            group = 1 if uid in running_users else 0
            return (group, tokens_served)
        
        return sorted(filtered, key=priority_key)
    
    def on_prefill_decision(self, batch):
        for req in batch.reqs:
            if req.uid:
                input_tokens = len(req.origin_input_ids) + len(req.output_ids)
                expected_output = req.sampling_params.max_new_tokens
                self.user_tokens_served[req.uid] = (
                    self.user_tokens_served.get(req.uid, 0)
                    + input_tokens
                    + expected_output
                )
    
    def on_decode_decision(self, batch):
        for req in batch.reqs:
            if req.uid:
                self.user_tokens_served[req.uid] = (
                    self.user_tokens_served.get(req.uid, 0) + 1
                )


class TimeSlicingPolicy(NoOpSchedulingPolicy):
    """Time-slice between prefill and decode to prevent prefill starvation.
    
    Alternates between prefill and decode passes based on a time window.
    Useful when continuous decode work prevents new requests from starting.
    """
    
    def __init__(self, prefill_window_ms=100, decode_window_ms=100):
        super().__init__()
        self.prefill_window_ms = prefill_window_ms
        self.decode_window_ms = decode_window_ms
        self.last_switch_time = 0
        self.current_mode = 'prefill'  # Start with prefill
        
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        """Alternate between prefill and decode based on time windows.
        
        Helper methods that could inform this decision:
        - len(waiting_queue) -> pending prefill work
        - len(running_batch.reqs) -> active decode work
        - new_prefill_batch is not None -> prefill work is ready
        """
        import time
        current_time = time.time() * 1000  # ms
        
        # No prefill work available -> must decode
        if new_prefill_batch is None:
            return None  # default (decode)
        
        # No decode work -> must prefill
        if running_batch.is_empty():
            return None  # default (prefill)
        
        # Check if we should switch modes
        time_in_mode = current_time - self.last_switch_time
        
        if self.current_mode == 'prefill':
            if time_in_mode >= self.prefill_window_ms:
                # Switch to decode
                self.current_mode = 'decode'
                self.last_switch_time = current_time
                return 'decode'
            else:
                return 'prefill'
        else:  # decode mode
            if time_in_mode >= self.decode_window_ms:
                # Switch to prefill
                self.current_mode = 'prefill'
                self.last_switch_time = current_time
                return 'prefill'
            else:
                return 'decode'


class StarvationPreventionPolicy(NoOpSchedulingPolicy):
    """Force decode when running requests have been waiting too long.
    
    Prevents continuous prefill from starving decode requests.
    """
    
    def __init__(self, max_decode_steps_without_progress=10):
        super().__init__()
        self.max_steps = max_decode_steps_without_progress
        self.request_last_decode_step = {}  # rid -> step counter
        self.global_step = 0
        
    def on_prefill_vs_decode_decision(self, waiting_queue, running_batch, new_prefill_batch):
        """Force decode if running requests haven't progressed recently.
        
        Helper methods that could inform this decision:
        - len(running_batch.reqs) -> how many requests need decode
        - Check req.uid on running_batch.reqs -> identify starved users
        - len(waiting_queue) -> how much prefill pressure exists
        """
        self.global_step += 1
        
        # No decode work -> allow prefill
        if running_batch.is_empty():
            return None
        
        # Check if any running request has been starved
        max_starvation = 0
        for req in running_batch.reqs:
            last_step = self.request_last_decode_step.get(req.rid, self.global_step)
            steps_since_decode = self.global_step - last_step
            max_starvation = max(max_starvation, steps_since_decode)
        
        # Force decode if any request is starved
        if max_starvation >= self.max_steps:
            return 'decode'
        
        return None  # default
    
    def on_decode_decision(self, batch):
        """Track when requests actually get decode steps."""
        for req in batch.reqs:
            self.request_last_decode_step[req.rid] = self.global_step


# Note: ResourceAwareFairnessPolicy requires scheduler reference
# See SCHEDULING_HOOK_DESIGN.md for the full implementation with scheduler helpers
