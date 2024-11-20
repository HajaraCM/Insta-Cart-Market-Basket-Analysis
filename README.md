# Instacart Market Basket Analysis and Product Reorder Prediction

This project analyzes 3 million anonymized grocery orders provided by Instacart to uncover customer purchasing patterns, product associations, and build predictive models for product reorders.The repository provides comprehensive insights into customer behavior, segmentation, and machine learning modeling.

![Screenshot 2024-11-20 153245](https://github.com/user-attachments/assets/145b4bc4-7e72-4a0e-8ec3-3689333ecefb)
![Screenshot 2024-11-20 153331](https://github.com/user-attachments/assets/a02b7e20-0003-4117-8e46-10d2feac0798)
![Screenshot 2024-11-20 153617](https://github.com/user-attachments/assets/a6ee5cbb-3753-4f16-87fb-c3c24de14cf2)
![Screenshot 2024-11-20 153702](https://github.com/user-attachments/assets/7cba88fb-2534-426e-8026-7880fcb6f808)



## Repository Structure
```
├── Plots/                                      : Contains all visualizations and plots 
├── Data Description and Analysis.ipynb         : Initial analysis to understand data
├── Exploratory Data Analysis.ipynb             : Detailed EDA for customer purchase patterns
├── Customers Segmentation.ipynb                : Customer segmentation based on product categories (aisles)
├── Market Basket Analysis.ipynb                : Analysis of product associations for cross-selling
├── Feature Extraction.ipynb                    : Feature engineering and extraction for ML modeling
├── Data Preparation.ipynb                      : Data preprocessing and preparation
├── Model.ipynb                                 : XGBoost model for predicting product reorders
├── final Model.ipynb                           : Final model combining results and showcasing performance 
├── LICENSE                                     : License for the project
└── README.md                                   : Project overview and instructions
```
<br>

## Objectives

1. **Analyze Instacart orders**: Uncover insights about customer purchasing patterns and reorder behavior.
2. **Market Basket Analysis**: Identify product associations for better cross-selling and upselling opportunities.
3. **Customer Segmentation**: Segment customers for targeted marketing based on product preferences.
4. **Product Reorder Prediction**: Build machine learning models to predict which previously purchased products will appear in a customer's next order.



## Data Description

The dataset includes information about:
- **Products**: 49,688 products categorized into 134 aisles and 21 departments.
- **Orders**: 3,421,083 orders from 206,209 customers, divided into `prior`, `train`, and `test` sets.
- **Order Details**: Information about products ordered, their order sequence, and reorder status.

### Key Observations:
- Customers typically order groceries weekly, with peaks at 7, 14, 21, and 30 days since prior orders.
- Most orders occur on weekends, with Saturday afternoons and Sunday mornings being the busiest.
- The majority of orders include 1–15 items, with a maximum of 145 items per order.
- Organic products, despite being fewer, have a higher reorder percentage.


## Analysis and Insights

### **Exploratory Data Analysis**
- **Popular Aisles and Departments**: Identified aisles like *fresh vegetables* and departments like *produce* as most popular.
- **Reorder Behavior**: Day-to-day items like fruits and vegetables have high reorder rates, unlike first-aid and beauty products.
- **Shelf Space Optimization**: 85% of purchases come from only 10,000 out of 49,688 products, indicating a potential for shelf space reduction.


### **Customer Segmentation**
Using **Principal Component Analysis (PCA)** and **KMeans Clustering**, customers were divided into 5 segments based on their purchase preferences:
1. **Water Enthusiasts**: Strong preference for water and sparkling water.
2. **Vegetable Buyers**: Primarily order fresh vegetables.
3. **Fruit Buyers**: Prefer fresh fruits and packaged produce.
4. **Balanced Shoppers**: Purchase both fruits and vegetables.
5. **New/Infrequent Users**: Order from many aisles but less frequently.


### **Market Basket Analysis**
**Association Rule Mining** with the Apriori Algorithm revealed product pairings with high lift values. For example:
- **Limes ↔ Large Lemons**: Lift = 3
- **Organic Strawberries ↔ Organic Raspberries**: Lift = 2.21

These insights can inform cross-selling strategies, store layout, and promotional campaigns.


### **Machine Learning Models**

#### Features Extracted:
1. **Product-Level Features**: Popularity, reorder rates, organic status, etc.
2. **Aisle/Department Features**: Reorder rates, user preferences, etc.
3. **User-Level Features**: Order frequency, purchase diversity, reorder behavior.
4. **User-Product Features**: Specific patterns for users reordering products.

#### XGBoost Model:
- **XGBoost** was chosen for its efficiency with large datasets and its ability to handle imbalanced classes.
- **Evaluation Metrics**: ROC-AUC was prioritized over F1 to avoid threshold manipulation.
- **Feature Importance**: XGBoost also provided insights into the most important features for predicting product reorders.

#### Results:
- The model performed well with an ROC-AUC score indicating its ability to predict reorder behaviors.
- Confusion matrices, classification reports, and ROC curves are shown to evaluate model performance.


### **Final Model**
The `final_model.ipynb` notebook combines the results and showcases the final model’s performance. It includes:
- End-to-end code to predict product reorders on new test data.
- Evaluation using Confusion Matrix, ROC-AUC, and Feature Importance analysis from XGBoost.
- Insights into the most important features driving product reorder predictions.


## How to Use the Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/HajaraCM/insta-cart-market-basket-analysis.git
   cd instacart-market-basket-analysis
   ```
2. Install required libraries:

```bash

pip install -r requirements.txt
```

3. Follow the notebooks for step-by-step analysis:

- Start with Data Description and Analysis.ipynb to understand the dataset.
- Explore customer behaviors in Exploratory Data Analysis.ipynb.
- Run Feature Extraction.ipynb and Data Preparation.ipynb to prepare the dataset for modeling.
- rain and evaluate the XGBoost model using XGBoost Model.ipynb.
- Check the complete results in final_model.ipynb.

### Results and Recommendations
- **Product Associations**: Use Market Basket Analysis findings to optimize product placement and cross-sell campaigns.
- **Customer Segmentation**: Implement targeted marketing strategies based on customer clusters.
- **Reorder Prediction**: Integrate the predictive model into Instacart's recommendation system to improve customer experience and increase sales.


