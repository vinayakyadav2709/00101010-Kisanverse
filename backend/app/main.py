import logging
import uvicorn
from fastapi import FastAPI
from routers import users, listings, contracts, subsidies, ai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Query, Response, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import httpx

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


@app.get("/proxy-image/")
async def proxy_image(image_url: str = Query(...)):
    appwrite_host = "https://d62b-124-66-175-40.ngrok-free.app"
    # Replace localhost URL with ngrok-hosted Appwrite URL
    proxied_url = image_url.replace("http://localhost", appwrite_host)
    logger.info(f"Original URL: {image_url}")
    logger.info(f"Proxied URL: {proxied_url}")

    headers = {
        "ngrok-skip-browser-warning": "true",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(proxied_url, headers=headers)
            logger.info(f"Response status: {resp.status_code}")
            if resp.status_code != 200:
                logger.error(f"Failed to fetch image: HTTP {resp.status_code}")
                raise HTTPException(
                    status_code=resp.status_code, detail="Image not found"
                )

            content_type = resp.headers.get("Content-Type", "image/jpeg")
            logger.info(f"Content-Type: {content_type}")

            # Log type of the content and first few bytes (limit to 100 bytes for readability)
            logger.info(f"Content type in Python: {type(resp.content)}")
            logger.info(f"First 100 bytes of content (raw): {resp.content[:100]!r}")

            return Response(content=resp.content, media_type=content_type)

    except Exception as e:
        logger.exception(f"Exception during image fetch: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
