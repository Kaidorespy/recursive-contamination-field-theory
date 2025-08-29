# Create a new file: phase6/echo_buffer.py

import numpy as np

class EchoBuffer:
    """Implements recursive echo buffer for memory protection"""
    
    def __init__(self, depth=0, decay_factor=0.8):
        """
        Initialize the echo buffer
        
        Parameters:
        -----------
        depth : int
            Depth of recursive echoing (0 = no protection)
        decay_factor : float
            How quickly echo strength decays with depth
        """
        self.depth = depth
        self.decay_factor = decay_factor
        self.buffer = []  # List of (state, strength) tuples
        
    def add_state(self, state):
        """Add a state to the echo buffer"""
        # Clear buffer if full AND not empty
        if self.buffer and len(self.buffer) >= self.depth:
            self.buffer.pop(0)  # Remove oldest entry
            
        # Add new state with full strength
        if self.depth > 0:
            self.buffer.append((state.copy(), 1.0))
            
    def apply_protection(self, current_state, intrusion_state, intrusion_strength):
        """
        Apply echo buffer protection against intrusion
        
        Parameters:
        -----------
        current_state : ndarray
            Current state being protected
        intrusion_state : ndarray
            Intrusion state being applied
        intrusion_strength : float
            Strength of intrusion
            
        Returns:
        --------
        ndarray
            Protected state
        """
        # If no buffer or empty buffer, no protection
        if self.depth == 0 or not self.buffer:
            # Apply intrusion directly
            return (1.0 - intrusion_strength) * current_state + intrusion_strength * intrusion_state
            
        # Apply recursive protection based on buffer depth
        protected_state = current_state.copy()
        
        # Calculate total echo strength (decreases with depth)
        total_echo_strength = sum(self.decay_factor ** i for i in range(len(self.buffer)))
        
        # Scale to ensure total influence is balanced
        echo_scale = (1.0 - intrusion_strength) / (1.0 + total_echo_strength)
        intrusion_scale = intrusion_strength
        
        # Apply intrusion and echoes
        result = echo_scale * protected_state
        
        # Add echo buffer reinforcement
        for i, (echo_state, echo_orig_strength) in enumerate(self.buffer):
            # Echo strength decreases with depth
            echo_strength = echo_orig_strength * (self.decay_factor ** i)
            result += echo_scale * echo_strength * echo_state
            
        # Add intrusion
        result += intrusion_scale * intrusion_state
        
        return result
