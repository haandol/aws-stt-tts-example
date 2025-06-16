from typing import Optional

from langchain_aws.chat_models import ChatBedrockConverse
from src.constant import MAX_TOKENS, TEMPERATURE
from src.logger import logger
from src.tools.adjust_light import adjust_light, get_brightness


class BedrockLLM(object):
    def __init__(
        self,
        model_id: str,
        aws_profile_name: Optional[str] = None,
    ):
        logger.info("🔍 BedrockLLM 초기화", model_id=model_id, aws_profile_name=aws_profile_name)
        self.model = ChatBedrockConverse(
            model_id=model_id,
            credentials_profile_name=aws_profile_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        ).bind_tools([get_brightness, adjust_light])
