from pydantic import BaseModel

class EmailRequest(BaseModel):
    subject: str
    body: str
