from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
import joblib


data = pd.read_csv('data_market.csv')
# Load the model
model = joblib.load('market.pkl')
# Load popular products from a pickle file
popular_data = pd.read_pickle('popular market')
#market basket from a pickle
market_basket=pd.read_pickle('market_basket')
# Filter to keep only the necessary columns
df = market_basket[['antecedents', 'consequents']]
print(df.head())


app = Flask(__name__)


product_images = {
    "Bag of Organic Bananas":"https://m.media-amazon.com/images/I/61gJG-6VTDL._SL1500_.jpg",
    "Banana": "https://m.media-amazon.com/images/I/71DuNPzDbmL._SL1500_.jpg",
    "Hass Avocado Variety": "https://m.media-amazon.com/images/I/51z3mQ7yhhL.jpg",
    "Gourmet Ole! Chicken Burritos": "https://m.media-amazon.com/images/I/813eUhMe8RL._SL1500_.jpg",
    "Organic Baby Spinach": "https://m.media-amazon.com/images/I/81O8q4qrSIL._SL1500_.jpg",
    "Chicken Tortilla Soup": "https://m.media-amazon.com/images/I/619Nvh3FyiL._SL1080_.jpg",
    "USDA AA Extra Large Eggs": "https://m.media-amazon.com/images/I/41++Gkb02ML.jpg",
    "Organic Hass Avocado":"https://m.media-amazon.com/images/I/51QEFmnhZRL.jpg",
    "Organic Strawberries":"https://m.media-amazon.com/images/I/61pdC8sjNfL._SL1314_.jpg",
    "Organic Avocado":"https://m.media-amazon.com/images/I/81LKLCmdAQL._SL1500_.jpg",
    "Large Lemon":"https://m.media-amazon.com/images/I/61lY-baJ9nL._SL1500_.jpg",
    "Strawberries":"https://m.media-amazon.com/images/I/81KdmBtYl5L._SL1500_.jpg",
    "Limes":"https://m.media-amazon.com/images/I/517FsBkm5bL.jpg",
    "Organic Whole Milk":"https://m.media-amazon.com/images/I/61dc7K2kmgL._SX300_SY300_QL70_FMwebp_.jpg",
    "Sparkling Lemon Water":"https://m.media-amazon.com/images/I/415jel1zC5L._SL1000_.jpg",
    "Sparkling Water Grapefruit":"https://m.media-amazon.com/images/I/71hDgkqoY+L._SL1500_.jpg",
    "Honeycrisp Apple":"https://m.media-amazon.com/images/I/61uOnnTXZZL._SL1227_.jpg",
    "Half & Half":"https://m.media-amazon.com/images/I/61rRk6265iL._SL1500_.jpg",
    "Organic Yellow Onion":"https://m.media-amazon.com/images/I/61-SuEyc-FL._AC_SL1016_.jpg",
    "Organic Garlic":"https://m.media-amazon.com/images/I/71KmgOL2q4L._SL1500_.jpg",
    "Organic Raspberries":"https://m.media-amazon.com/images/I/61vq3e1Rf0L._SL1000_.jpg"
}


def get_user_orders(user_id):
    return data[data['user_id'] == user_id]

def generate_features(user_orders):
    if user_orders.empty:
        return None
    relevant_features = user_orders.drop(columns=['user_id', 'product_id', 'product_name', 'reordered']).iloc[-1:]
    return relevant_features

def predict_reorder(user_features):
    if user_features is None:
        return None, "No previous orders found for this user."
    dmatrix = xgb.DMatrix(user_features)
    probabilities = model.predict(dmatrix)
    predictions = [1 if i > 0.5 else 0 for i in probabilities]
    return probabilities, predictions

def suggest_products(user_orders, predictions):
    if not any(predictions):
        return []
    
    reordered_products = user_orders[user_orders['reordered'] == 1]
    suggested_products = reordered_products['product_name'].unique().tolist()
    
    return suggested_products



df['antecedents'] = df['antecedents'].apply(lambda x: tuple(item.strip() for item in x))
df['consequents'] = df['consequents'].apply(lambda x: tuple(item.strip() for item in x))


# Main page route
@app.route('/')
def home():
    results = {} 
    return render_template('index.html',results=results)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    results = {}
    if request.method == 'POST':
        user_id = request.form.get('user_id', type=int)
        
        if user_id is None:
            results['error'] = "User ID is required."
        else:
            user_orders = get_user_orders(user_id)
            if user_orders.empty:
                results['error'] = "No orders found for this user."
            else:
                # Generate features and make predictions
                user_features = generate_features(user_orders)
                probabilities, predictions = predict_reorder(user_features)
                
                # Suggest products based on predictions
                suggested_products = suggest_products(user_orders, predictions)
                
                # Store results to display
                results['previous_product_names'] =  user_orders['product_name'].tolist()
                results['suggested_products'] = suggested_products
    
    return render_template('predict.html', results=results, product_images=product_images)

@app.route('/popular', methods=['GET'])
def popular():

    popular_products= popular_data['product_name'].tolist()
 
    return render_template('popular.html', popular_products=popular_products, product_images=product_images)

@app.route('/basket', methods=['GET','POST'])
def basket():
    recommendations = {}
    product_name = ""
    if request.method == 'POST':
        product_name = request.form['product-name'].strip()  # Normalize user input

        # Find matches for antecedents and consequents
        matches_as_antecedent = df[df['antecedents'].apply(lambda x: product_name in x)]
       
        # Aggregate unique consequent products
        unique_recommendations = set()
        
        # Collect recommendations from antecedents
        for _, row in matches_as_antecedent.iterrows():
            unique_recommendations.update(row['consequents'])
        
       

        # Build recommendations dictionary with images
        for item in unique_recommendations:
            recommendations[item] = product_images.get(item, None)

    return render_template('basket.html', recommendations=recommendations, product_name=product_name)


if __name__ == '__main__':
    app.run(debug=True)
