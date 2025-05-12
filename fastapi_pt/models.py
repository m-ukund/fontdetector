from pydantic import BaseModel
from typing import Optional

class Feedback(BaseModel):
    is_canary: bool
    is_correct: bool
    latency: float
    feedback_text: Optional[str] = None 