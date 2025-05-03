# import logging
# import uvicorn
# from fastapi import FastAPI
# from routers import users, listings, contracts, subsidies, ai
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi import Request
# from starlette.middleware.base import BaseHTTPMiddleware
# import traceback

# # run subisides test function

# # subsidies.run_tests()
# # contracts.run_tests()
# # upload.init()
# # ai.tests()
# ai.test_test()
from routers import insert_weather
# from routers import create_mapping

# create_mapping.insert_zipcodes()
# insert_weather.fetch_weather_data()
from routers import fetch_data

fetch_data.fetch_and_store_weather_data()
