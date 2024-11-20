# Instacart Market Basket Analysis and Product Reorder Prediction

This project analyzes 3 million anonymized grocery orders provided by Instacart to uncover customer purchasing patterns, product associations, and build predictive models for product reorders.The repository provides comprehensive insights into customer behavior, segmentation, and machine learning modeling.


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
---

## Objectives

1. **Analyze Instacart orders**: Uncover insights about customer purchasing patterns and reorder behavior.
2. **Market Basket Analysis**: Identify product associations for better cross-selling and upselling opportunities.
3. **Customer Segmentation**: Segment customers for targeted marketing based on product preferences.
4. **Product Reorder Prediction**: Build machine learning models to predict which previously purchased products will appear in a customer's next order.

---

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

---

## Analysis and Insights

### **Exploratory Data Analysis**
- **Popular Aisles and Departments**: Identified aisles like *fresh vegetables* and departments like *produce* as most popular.
- **Reorder Behavior**: Day-to-day items like fruits and vegetables have high reorder rates, unlike first-aid and beauty products.
- **Shelf Space Optimization**: 85% of purchases come from only 10,000 out of 49,688 products, indicating a potential for shelf space reduction.

---

### **Customer Segmentation**
Using **Principal Component Analysis (PCA)** and **KMeans Clustering**, customers were divided into 5 segments based on their purchase preferences:
1. **Water Enthusiasts**: Strong preference for water and sparkling water.
2. **Vegetable Buyers**: Primarily order fresh vegetables.
3. **Fruit Buyers**: Prefer fresh fruits and packaged produce.
4. **Balanced Shoppers**: Purchase both fruits and vegetables.
5. **New/Infrequent Users**: Order from many aisles but less frequently.

---

### **Market Basket Analysis**
**Association Rule Mining** with the Apriori Algorithm revealed product pairings with high lift values. For example:
- **Limes ↔ Large Lemons**: Lift = 3
- **Organic Strawberries ↔ Organic Raspberries**: Lift = 2.21

These insights can inform cross-selling strategies, store layout, and promotional campaigns.

---

### **Machine Learning Models**

#### Features Extracted:
1. **Product-Level Features**: Popularity, reorder rates, organic status, etc.
2. **Aisle/Department Features**: Reorder rates, user preferences, etc.
3. **User-Level Features**: Order frequency, purchase diversity, reorder behavior.
4. **User-Product Features**: Specific patterns for users reordering products.

#### Models:
- **XGBoost**: Used for its robustness with large datasets and ability to handle imbalanced classes. Feature importance analysis was also performed.
- **Artificial Neural Network (ANN)**: Explored for potential performance gains with large-scale data.

#### Results:
- **Evaluation Metrics**: ROC-AUC was prioritized over F1 to avoid threshold manipulation.
- Both models performed similarly, with **XGBoost slightly outperforming ANN**.

---

### **Final Model**
The `final_model.ipynb` notebook combines the best insights and results from the XGBoost and ANN models. It showcases:
- Confusion matrix, classification report, and ROC curves.
- Feature importance derived from XGBoost.
- End-to-end flow for predicting product reorders on new test data.

---

## How to Use the Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/instacart-market-basket-analysis.git
   cd instacart-market-basket-analysis


