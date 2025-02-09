from pydantic import BaseModel

class DraftRequest(BaseModel):
    theme: str
    urgency: str = "normal"  # low/normal/high

class DraftEdit(BaseModel):
    text: str
    apply_corrections: bool = True