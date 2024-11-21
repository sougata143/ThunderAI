from fastapi import APIRouter, Depends
from typing import Dict, Any
from core.cache import CacheManager
from db.models import Prediction
from db.crud import prediction

router = APIRouter() 