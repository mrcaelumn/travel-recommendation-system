from typing import List, Dict
from pydantic import BaseModel

class UserInput(BaseModel):
    user_id: int
    top_n: int = 10
    ratings: list[int]