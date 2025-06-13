import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # noqa: E402

import structlog


logger = structlog.get_logger("config")

# AWS
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", None)
logger.info("AWS region configuration", region=AWS_DEFAULT_REGION)
AWS_PROFILE = os.getenv("AWS_PROFILE", None)
logger.info("AWS profile configuration", profile=AWS_PROFILE)
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
logger.info("Model configuration", model_id=MODEL_ID)

# Transcribe
LANG_CODE = os.environ.get("LANG_CODE", "ko-KR")
logger.info(f"LANG_CODE: {LANG_CODE}")

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
logger.info("Environment configuration", environment=ENVIRONMENT)

# TTS
VOICE_ID = os.environ.get("VOICE_ID", "Seoyeon")  # 지혜 목소리
logger.info(f"VOICE_ID: {VOICE_ID}")

# Porcupine
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY")
if not PORCUPINE_ACCESS_KEY:
    logger.warning("Porcupine access key is not set. Wake word detection will not work.")
logger.info("Porcupine configuration", has_access_key=bool(PORCUPINE_ACCESS_KEY))
PORCUPINE_WAKE_WORD = os.environ.get("PORCUPINE_WAKE_WORD", "jarvis")
logger.info(f"PORCUPINE_WAKE_WORD: {PORCUPINE_WAKE_WORD}")


@dataclass
class Config:
    aws_default_region: Optional[str]
    aws_profile: Optional[str]
    model_id: str
    lang_code: str
    environment: str
    voice_id: str
    porcupine_access_key: Optional[str]
    wake_word: str


config = Config(
    aws_default_region=AWS_DEFAULT_REGION,
    aws_profile=AWS_PROFILE,
    model_id=MODEL_ID,
    lang_code=LANG_CODE,
    environment=ENVIRONMENT,
    voice_id=VOICE_ID,
    porcupine_access_key=PORCUPINE_ACCESS_KEY,
    wake_word=PORCUPINE_WAKE_WORD,
)
