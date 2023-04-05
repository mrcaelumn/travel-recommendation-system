from fastapi import APIRouter
from api.web.api.predictions.schema import UserInput
from api.web.api.predictions.models import OurModel

router = APIRouter()
models = OurModel()

@router.post("/recommendations/als")
async def get_als_recommendations(user_input: UserInput):
    
    recommended_items = models.predict_als(user_input.user_id, user_input.top_n) 
    
    # Return the recommended items
    return {"recommended_items": recommended_items}

@router.post("/recommendations/rbm")
async def get_rbm_recommendations(user_input: UserInput):
    
    recommended_items = models.predict_rbm(user_input.ratings, user_input.top_n) 
    # print("recommended_items:", recommended_items.tolist())
    # Return the recommended items
    return {"recommended_items": recommended_items.tolist()}
