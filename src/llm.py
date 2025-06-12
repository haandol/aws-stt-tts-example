import logging
from typing import Optional

from langchain_aws.chat_models import ChatBedrockConverse
from src.constant import MAX_TOKENS, TEMPERATURE


class BedrockLLM(object):
    def __init__(
        self,
        model_id: str,
        aws_profile_name: Optional[str] = None,
    ):
        self.model = ChatBedrockConverse(
            model_id=model_id,
            credentials_profile_name=aws_profile_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
