import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ExecutionTraceStep(BaseModel):
    """
    Represents a single step in the observable execution trace.
    This schema is strictly enforced for evaluation scoring.
    """
    step_id: str = Field(..., description="Unique identifier for the execution step")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the step")
    module: str = Field(..., description="The module performing the action (e.g., 'encoder', 'decoder', 'evaluator')")
    action: str = Field(..., description="The action being performed (e.g., 'extract_features', 'generate_text')")
    
    # Inputs and outputs should be JSON serializable dicts
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Inputs provided to the step")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs produced by the step")
    
    # Any additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or metadata")

class ExecutionTrace(BaseModel):
    """
    Represents the full execution trace for a run.
    """
    run_id: str = Field(..., description="Unique identifier for the entire run")
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    steps: List[ExecutionTraceStep] = Field(default_factory=list)
    final_metrics: Dict[str, float] = Field(default_factory=dict, description="Aggregated final metrics")

    def add_step(self, step: ExecutionTraceStep):
        self.steps.append(step)

    def to_json(self) -> str:
        """Serialize the trace to a strict JSON string."""
        # Using model_dump_json for Pydantic v2
        return self.model_dump_json(indent=2)
    
    def save(self, filepath: str):
        """Save the trace to a file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
