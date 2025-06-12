import boto3
import io
import asyncio
from typing import Optional
import pygame
from src.logger import logger
from src.config import config


class PollyTTS:
    def __init__(self, aws_profile_name: Optional[str] = None):
        """
        Amazon Polly를 사용한 TTS 클래스 초기화

        Args:
            aws_profile_name: AWS 프로파일 이름 (선택사항)
        """
        logger.info("🔊 Polly TTS 초기화", aws_profile_name=aws_profile_name)

        # AWS 세션 생성
        if aws_profile_name:
            session = boto3.Session(profile_name=aws_profile_name)
        else:
            session = boto3.Session()

        # Polly 클라이언트 생성
        self.polly_client = session.client("polly", region_name=config.aws_default_region or "us-east-1")

        # pygame mixer 초기화 (오디오 재생용)
        pygame.mixer.init()

        logger.info("✅ Polly TTS가 초기화되었습니다.")

    async def speak_async(self, text: str, voice_id: str) -> None:
        """
        비동기적으로 텍스트를 음성으로 변환하고 재생

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID (기본값: config에서 설정된 값 사용)
        """
        try:
            logger.info("🔊 음성 변환 시작", text=text[:50] + "..." if len(text) > 50 else text, voice_id=voice_id)

            # 비동기 처리를 위해 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._synthesize_and_play, text, voice_id)

        except Exception as e:
            logger.error(f"❌ TTS 오류 발생: {e}")

    def _synthesize_and_play(self, text: str, voice_id: str) -> None:
        """
        텍스트를 음성으로 변환하고 재생하는 내부 메서드

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID
        """
        try:
            # Polly로 음성 합성
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=voice_id,
                LanguageCode="ko-KR",  # 한국어
                Engine="neural",  # 더 자연스러운 음성을 위해 neural 엔진 사용
            )

            # 오디오 스트림 읽기
            audio_stream = response["AudioStream"].read()

            # 메모리에서 오디오 재생
            audio_file = io.BytesIO(audio_stream)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # 재생 완료까지 대기
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            logger.info("✅ 음성 재생 완료")

        except Exception as e:
            logger.error(f"❌ 음성 합성 또는 재생 오류: {e}")

    def speak_sync(self, text: str, voice_id: str = None) -> None:
        """
        동기적으로 텍스트를 음성으로 변환하고 재생

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID
        """
        if voice_id is None:
            voice_id = config.voice_id
        self._synthesize_and_play(text, voice_id)

    def get_available_voices(self) -> list:
        """
        사용 가능한 한국어 음성 목록 반환

        Returns:
            한국어 음성 목록
        """
        try:
            response = self.polly_client.describe_voices(LanguageCode="ko-KR")
            voices = response["Voices"]

            logger.info("🎵 사용 가능한 한국어 음성 목록:")
            for voice in voices:
                logger.info(f"  - {voice['Name']} ({voice['Id']}) - {voice['Gender']}")

            return voices

        except Exception as e:
            logger.error(f"❌ 음성 목록 조회 오류: {e}")
            return []
