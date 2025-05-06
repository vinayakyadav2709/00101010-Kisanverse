# run_api_example.py
import recommendation_api # Prediction Function
import json
import logging # Optional: Configure logging for the caller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Option 1: Call with file path
input_json_path = "ss.json"
print(f"Getting recommendations from file: {input_json_path}")
recommendation_result = recommendation_api.get_recommendations(input_json_path)

# Option 2: Call with dictionary
# print("\nGetting recommendations from dictionary...")
# try:
#     with open("ss.json", 'r') as f:
#         input_dict = json.load(f)
#     recommendation_result = recommendation_api.get_recommendations(input_dict)
# except Exception as e:
#      print(f"Error preparing dictionary input: {e}")
#      recommendation_result = None


# Process the result
if recommendation_result:
    print("\n--- Recommendation Received ---")
    # You can now work with the dictionary
    # print(json.dumps(recommendation_result, indent=2)) # Pretty print the full result
    print(f"Summary: {recommendation_result}")
else:
    print("\n--- Failed to get recommendations ---")
    print("Check the logs in the console output for details.")