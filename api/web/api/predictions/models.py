import tensorflow as tf
from api.models_config import CONFIG

from models.rbm.rbm_model import RBM
from pyspark.ml.recommendation import ALSModel
import pandas as pd
import numpy as np
import findspark
from pyspark.sql import SparkSession
from sklearn.preprocessing import StandardScaler

findspark.init()
    
spark = SparkSession.builder.master("local[*]").getOrCreate()

class OurModel:
    def __init__(self):
        
        # declare the models here
        # You can then load the model using the following code:
        loaded_checkpoint = tf.train.Checkpoint(model=RBM(CONFIG["RBM_VISIBLE_UNITS"], CONFIG["RBM_HIDDEN_UNITS"]))
        loaded_checkpoint.restore(CONFIG["RBM_MODEL_PATH"])
        
        self.rbm_model = loaded_checkpoint.model
        
        self.als_model = ALSModel.load(CONFIG["ALS_MODEL_PATH"])

    def predict_als(self, user_id, top_n=10):
        
        # Use the loaded ALS model to generate hotel recommendations for the given user
        recommendations = self.als_model.recommendForAllUsers(top_n)
        
        # Filter recommendations for the given user and return the top 5 recommended hotels
        user_recommendations = recommendations.filter(recommendations.user_id == user_id).select("recommendations").first()[0]
        top_recommendations = [r.hotel_id for r in user_recommendations][:top_n]
        
        
        return top_recommendations
        
    def predict_rbm(self, user_ratings, top_n=10):
        # print("user rating", user_ratings)
        # Convert the user's ratings to a numpy array
        ratings_array = np.array(user_ratings).reshape(1, -1)

        # Create a dummy input array for the visible units
        visible_units = np.zeros((1, CONFIG["RBM_VISIBLE_UNITS"]))

        # Copy the user's ratings into the visible units array
        visible_units[:, :len(user_ratings)] = ratings_array

        # Normalize the training data to have zero mean and unit variance

        scaler = StandardScaler()
        user_ratings_final = scaler.fit_transform(visible_units)
        # print("user_ratings_final", user_ratings_final)
        hidden_representation = self.rbm_model.sample_hidden(tf.constant([user_ratings_final], dtype=tf.float32))
        predicted_ratings = self.rbm_model.sample_visible(hidden_representation)
        recommended_items = (-predicted_ratings.numpy()).argsort()[0]
        final_recoms = recommended_items[0][:top_n]
        
        return final_recoms

        
    
    





