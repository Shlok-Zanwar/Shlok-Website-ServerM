from fastapi import APIRouter, HTTPException, status

router = APIRouter(
    tags=['Blogs'],
)


blogChoice = {
    "dock-forms": "./BlogJsons/DockFormsBlog.json",
    "todo-app": "./BlogJsons/TodoBlog.json",
    "toxicbot": "./BlogJsons/ToxicbotBlog.json",
    "hotel-management": "./BlogJsons/HotelManagementBlog.json"
}


@router.get('/blogs/{blogName}')
def dockForms(blogName: str):
    # time.sleep(2)
    try:
        file = open(blogChoice[blogName], 'r+')
        return ({"blogDetail": file.read()})
    except:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


