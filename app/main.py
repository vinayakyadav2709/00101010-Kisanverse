from fastapi import FastAPI
from routers import users, listings, contracts, subsidies
import uvicorn

app = FastAPI()

routers = [
    users.router,
    listings.listings_router,
    listings.bids_router,
    contracts.contracts_router,
    contracts.contract_requests_router,
    subsidies.subsidies_router,
    subsidies.subsidy_requests_router,
]

for router in routers:
    app.include_router(router)


def run_tests():
    users.run_tests()
    listings.run_tests()
    contracts.run_tests()
    subsidies.run_tests()


# Uncomment the following line to run tests
# run_tests()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
