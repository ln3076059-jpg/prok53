from __future__ import annotations

import argparse

from sqlalchemy import select

from backend.core.security import hash_password
from backend.database import Base, SessionLocal, engine
from backend.models.entities import User


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("password")
    args = parser.parse_args()
    if len(args.password) < 12:
        raise ValueError("admin password must contain at least 12 characters")
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        if session.scalar(select(User).where(User.email == args.email.lower())):
            raise ValueError("user already exists")
        session.add(User(email=args.email.lower(), password_hash=hash_password(args.password), role="admin"))
        session.commit()
    print("Admin created")


if __name__ == "__main__":
    main()

