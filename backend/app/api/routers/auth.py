"""
用户认证模块。

负责：

1. 用户注册
2. 用户登录
3. JWT Token生成
4. 当前用户身份获取
5. 团支书权限验证


核心流程：

用户名密码
    ↓
密码Hash验证
    ↓
生成JWT
    ↓
前端保存Token
    ↓
访问其他接口


用于保护：

- 知识库
- 会议管理
- 管理员功能
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import create_token, get_current_user, hash_password, require_secretary, verify_password
from app.models.entities import ClassRoom, User


router = APIRouter(prefix="/api/auth", tags=["账号"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=6, max_length=100)
    display_name: str = Field(min_length=1, max_length=80)
    invite_code: str


def user_payload(user: User, classroom: ClassRoom) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "class_id": user.class_id,
        "class_name": classroom.name,
    }


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)) -> dict:
    user = db.scalar(select(User).where(User.username == request.username.strip()))
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="账号或密码错误")
    classroom = db.get(ClassRoom, user.class_id)
    return {"access_token": create_token(user), "token_type": "bearer", "user": user_payload(user, classroom)}


@router.post("/register", status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    if db.scalar(select(User).where(User.username == request.username.strip())):
        raise HTTPException(status_code=409, detail="账号已经存在")
    classroom = db.scalar(select(ClassRoom).where(ClassRoom.invite_code == request.invite_code.strip()))
    if not classroom:
        raise HTTPException(status_code=400, detail="班级邀请码无效")
    user = User(
        username=request.username.strip(),
        password_hash=hash_password(request.password),
        display_name=request.display_name.strip(),
        role="student",
        class_id=classroom.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": create_token(user), "token_type": "bearer", "user": user_payload(user, classroom)}


@router.get("/me")
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return user_payload(user, db.get(ClassRoom, user.class_id))


@router.get("/class-invite")
def class_invite(user: User = Depends(require_secretary), db: Session = Depends(get_db)) -> dict:
    classroom = db.get(ClassRoom, user.class_id)
    return {"class_name": classroom.name, "invite_code": classroom.invite_code}
