from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the recommendation model
model = load_model('recommendation_model.h5')

# Load restaurant data
restaurant_data = pd.read_csv('restaurant.csv')

# Load business data
business_data = pd.read_csv('/Users/khushpatel/Desktop/Recommendation /Data Preprocessing/modified_business.csv')

# Function for making predictions
def predict_recommendations_for_new_user(new_user_data):
    # Extract relevant features for content-based filtering
    content_based_columns = ['American (New)', 'American (Traditional)', 
                              'Asian Fusion', 'Bagels', 'Bakeries', 'Barbeque', 'Bars', 'Bubble Tea', 'Burgers', 
                              'Caribbean', 'Chicken Wings', 'Chinese', 'Coffee & Tea', 'Comfort Food', 'Dance Clubs', 
                              'Desserts', 'Diners', 'Donuts', 'Ethnic Food', 'Fast Food', 'French', 'Gluten-Free', 
                              'Greek', 'Halal', 'Ice Cream & Frozen Yogurt', 'Indian', 'Irish', 'Italian', 'Japanese', 
                              'Juice Bars & Smoothies', 'Korean','Lounges', 'Mediterranean', 'Mexican', 
                              'Middle Eastern', 'Nightlife', 'Noodles', 'Pizza', 'Ramen', 'Salad', 'Sandwiches', 'Seafood', 
                              'Soup', 'Southern', 'Spanish', 'Steakhouses', 'Sushi Bars', 'Szechuan', 'Tacos', 'Tapas Bars', 
                              'Tapas/Small Plates', 'Tex-Mex', 'Thai', 'Vegan', 'Vegetarian', 'Vietnamese']
    
    # Check if the specified columns exist in new_user_data
    missing_columns_user = set(content_based_columns) - set(new_user_data.columns)
    
    if missing_columns_user:
        raise KeyError(f"Columns {missing_columns_user} not found in new_user_data.")
    
    # Check if the specified columns exist in restaurant_data
    missing_columns_restaurant = set(content_based_columns) - set(restaurant_data.columns)
    
    if missing_columns_restaurant:
        raise KeyError(f"Columns {missing_columns_restaurant} not found in restaurant_data.")
    
    # Calculate the similarity between new user features and restaurant features
    similarity_scores = np.dot(restaurant_data[content_based_columns].values, new_user_data[content_based_columns].values.flatten())
    
    # Get the indices of top_n most similar businesses
    top_business_indices = np.argsort(similarity_scores)[::-1][:10]
    
    # Get the business_ids of the top_n most similar businesses
    recommended_businesses = restaurant_data.loc[top_business_indices, 'business_id']
    
    # Merge with business_data to get additional information
    recommended_businesses = pd.merge(recommended_businesses, business_data, on='business_id', how='left')
    
    return recommended_businesses[['business_id', 'name', 'categories', 'stars']]

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify()
    else:
        data = request.json
        new_user_data = pd.DataFrame(data, index=[0])
        
        # Make predictions
        recommended_businesses = predict_recommendations_for_new_user(new_user_data)
        
        response = jsonify(recommended_businesses.to_dict(orient='records'))
    
    # Add CORS headers to the response
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    
    return response

@app.route("/")
def index():
            return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)

