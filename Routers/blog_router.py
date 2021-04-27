from fastapi import APIRouter, Depends, Request
import json
import time

router = APIRouter(
    tags=['Blogs'],
    prefix="/blogs"
)

@router.get('/dock-forms')
def dockForms():
    file = open('./BlogJsons/DockFormsBlog.json', 'r+')
    return ({"blogDetail": file.read()})

@router.get('/todo-app')
def todoApp():
    file = open('./BlogJsons/TodoBlog.json', 'r+')
    return ({"blogDetail": file.read()})

@router.get('/toxicbot')
def toxicbot():
    file = open('./BlogJsons/ToxicbotBlog.json', 'r+')
    return ({"blogDetail": file.read()})

@router.get('/hotel-managment')
def hotelManagement():
    file = open('./BlogJsons/HotelManagementBlog.json', 'r+')
    return ({"blogDetail": file.read()})