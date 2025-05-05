# import logging
# import uvicorn
# from fastapi import FastAPI
# from routers.ai import test_price_apis_and_crop_prediction

# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi import Request
# from starlette.middleware.base import BaseHTTPMiddleware
# import traceback
# test_price_apis_and_crop_prediction()
# # run subisides test function
# test_price_apis_and_crop_prediction()
# # subsidies.run_tests()
# # contracts.run_tests()
# # upload.init()
# # ai.tests()
# ai.test_test()
# from routers.other_scripts.insert_weather import adjust_weather_data, printt
# from routers.other_scripts.fetch_data import (
#     count_unique_lat_lon,
#     fetch_and_store_weather_data,
# )
# from routers.other_scripts.upload_prices import (
#     insert_prices_from_files,
#     update_crop_names_to_uppercase,
# )
# fetch_and_store_weather_data()
# count_unique_lat_lon()
# adjust_weather_data()
# printt()

# insert_prices_from_files()
# update_crop_names_to_uppercase()
# from routers.other_scripts.approx import main

# main()

# from models.llm import test




from routers.other_scripts.days import fetch_and_store_weather_data

fetch_and_store_weather_data(21.5929 , 81.3761)
