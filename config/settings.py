from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str
    SUPERTEAM_DATA_DIR: str = "./superteam_data"
    ADMIN_TOKENS: list[str] = []
    USE_LOCAL_LLM: bool = False
    PUBLISH_MODE: str = "simulation"  # simulation or production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self):
        super().__init__()
        self.ADMIN_TOKENS = os.getenv("ADMIN_TOKENS", "").split(",")

config = Settings()