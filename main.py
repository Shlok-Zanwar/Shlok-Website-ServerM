from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Routers import blog_router
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


app.include_router(blog_router.router)
# app.include_router(ml_forum_model.router)
# app.include_router(socket_router.router)
# app.include_router(model_router.router)

# if __name__ == '__main__':
#     uvicorn.run("main:app", host="localhost", port=8000, reload=True)
