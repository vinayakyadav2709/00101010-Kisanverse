import logging
import uvicorn
from fastapi import FastAPI
from routers import users, listings, contracts, subsidies, ai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(debug=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log detailed error information
    logger.error(f"Unhandled exception: {traceback.format_exc()}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Request headers: {request.headers}")

    # Log request body if applicable
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            logger.error(f"Request body: {body.decode('utf-8')}")
        except Exception as body_exc:
            logger.error(f"Failed to read request body: {str(body_exc)}")

    # Return a generic error response
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Log request details
        logger.info(f"Request: {request.method} {request.url}")
        logger.info(f"Headers: {request.headers}")
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            # Log the body only if it's not binary data
            if "multipart/form-data" not in request.headers.get("content-type", ""):
                logger.info(f"Body: {body.decode('utf-8')}")
            else:
                logger.info("Binary data received, skipping body logging.")

        # Process the request
        response = await call_next(request)

        # Log response details
        logger.info(f"Response status: {response.status_code}")
        return response


app.add_middleware(LogMiddleware)

routers = [
    users.router,
    listings.listings_router,
    listings.bids_router,
    contracts.contracts_router,
    contracts.contract_requests_router,
    subsidies.subsidies_router,
    subsidies.subsidy_requests_router,
    ai.ai_router,
]

for router in routers:
    app.include_router(router)

if __name__ == "__main__":
    # Run uvicorn with debug-level logging
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
