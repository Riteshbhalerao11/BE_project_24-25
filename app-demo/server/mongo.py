import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB URI from .env
MONGODB_URI = os.getenv("MONGODB_URI")
print(MONGODB_URI)

# Connect to MongoDB
client = MongoClient(MONGODB_URI)

# Select database and collection
db = client["test"]  # Replace with your database name
products_collection = db["product2"]  # Replace with your collection name

# Function to get product by ID
def get_product_by_id(product_id):
    try:
        # Fetch the product by its ObjectId
        product = products_collection.find_one({"_id": ObjectId(product_id)})
        if not product:
            return None

        return product
    except Exception as e:
        print(f"Error fetching product: {e}")
        return None

# Example usage
# product_id = "670d1cae9884470f8e978c87"  # Replace with an actual ObjectId
# product = get_product_by_id(product_id)
# if product:
#     print(f"Product Description: {product.get('description')}")
# else:
#     print("Product not found")
