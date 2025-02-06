# models.py
from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    telegram_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    # Preferences can be stored as a JSON string
    preferences = Column(Text, nullable=True)

    def set_preferences(self, prefs: dict):
        self.preferences = json.dumps(prefs)

    def get_preferences(self) -> dict:
        return json.loads(self.preferences) if self.preferences else {}

# Create an SQLite database for simplicity. For production, consider PostgreSQL.
engine = create_engine("sqlite:///./users.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)
