from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from med_assistant.api.deps import get_current_user
from med_assistant.db.database import get_db
from med_assistant.models.schemas import AuthResponse, UserLogin, UserRead, UserSignup
from med_assistant.models.user import User
from med_assistant.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_user,
    get_user_by_email,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: UserSignup, db: Session = Depends(get_db)):
    email = payload.email.lower()
    if get_user_by_email(db, email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = create_user(db, email, payload.password)
    token = create_access_token(user.email)
    return AuthResponse(access_token=token, user=UserRead(email=user.email))


@router.post("/login", response_model=AuthResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token(user.email)
    return AuthResponse(access_token=token, user=UserRead(email=user.email))


@router.post("/logout")
def logout(_: User = Depends(get_current_user)):
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return UserRead(email=current_user.email)
