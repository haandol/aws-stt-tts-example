from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
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
        model = ChatBedrockConverse(
            model_id=model_id,
            credentials_profile_name=aws_profile_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        system_message = SystemMessage(
            content="당신은 유용한 AI 비서입니다. 모든 답변은 사용자의 언어로 답변해 주세요."
        )
        checkpointer = InMemorySaver()
        self.model = create_react_agent(
            model=model,
            prompt=system_message,
            checkpointer=checkpointer,
            tools=[get_brightness, adjust_light],
        )
