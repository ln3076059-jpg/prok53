from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.security import decode_access_token
from backend.database import get_db
from backend.models.entities import User

bearer = HTTPBearer(auto_error=False)


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if not credentials or not (email := decode_access_token(credentials.credentials)):
        raise HTTPException(status_code=401, detail="Authentication required")
    user = db.scalar(select(User).where(User.email == email))
    if not user:
        raise HTTPException(status_code=401, detail="Unknown user")
    return user


def require_roles(*allowed_roles: str):
    """Build a FastAPI dependency that enforces an explicit role allow-list."""
    allowed = frozenset(role.strip().lower() for role in allowed_roles)
    if not allowed:
        raise ValueError("at least one role is required")

    def authorize(user: User = Depends(current_user)) -> User:
        if (user.role or "").strip().lower() not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return authorize


reviewer_user = require_roles("reviewer", "admin")
export_user = require_roles("reviewer", "auditor", "admin")
admin_user = require_roles("admin")
