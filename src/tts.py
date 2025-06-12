import asyncio
from typing import Optional

import boto3
import numpy as np
import sounddevice as sd

from src.config import config
from src.logger import logger


class PollyTTS:
    def __init__(self, voice_id: str, aws_profile: Optional[str] = None):
        """
        Amazon Polly를 사용한 TTS 클래스 초기화

        Args:
            voice_id: 사용할 음성 ID
            aws_profile_name: AWS 프로파일 이름 (선택사항)
        """
        logger.info("🔊 Polly TTS 초기화", voice_id=voice_id, aws_profile=aws_profile)

        # AWS 세션 생성
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
        else:
            session = boto3.Session(region_name=config.aws_default_region)

        # Polly 클라이언트 생성
        self.polly_client = session.client("polly")
        self.voice_id = voice_id

        logger.info("✅ Polly TTS가 초기화되었습니다.")

    async def speak_async(self, text: str) -> None:
        """
        비동기적으로 텍스트를 음성으로 변환하고 재생

        Args:
            text: 변환할 텍스트
        """
        try:
            logger.info("🔊 음성 변환 시작", text=text[:50] + "..." if len(text) > 50 else text)

            # 비동기 처리를 위해 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._synthesize_and_play, text)

        except Exception as e:
            logger.error(f"❌ TTS 오류 발생: {e}")

    def _synthesize_and_play(self, text: str) -> None:
        """
        텍스트를 음성으로 변환하고 재생하는 내부 메서드

        Args:
            text: 변환할 텍스트
        """
        try:
            # Polly로 음성 합성 (PCM 포맷으로 직접 출력)
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat="pcm",  # PCM 포맷으로 변경
                VoiceId=self.voice_id,
                LanguageCode="ko-KR",  # 한국어
                Engine="neural",  # 더 자연스러운 음성을 위해 neural 엔진 사용
                SampleRate="16000",  # PCM용 샘플레이트
            )

            # 오디오 스트림 읽기
            audio_stream = response["AudioStream"].read()

            # PCM 데이터를 NumPy 배열로 직접 변환
            # Polly PCM은 16-bit signed integer, mono, little-endian
            audio_data = np.frombuffer(audio_stream, dtype=np.int16)

            # float32로 정규화 (-1.0 ~ 1.0 범위)
            audio_data = audio_data.astype(np.float32) / 32768.0

            # sounddevice로 직접 재생 (메모리 스트리밍)
            sd.play(audio_data, samplerate=16000)
            sd.wait()  # 재생 완료까지 대기

            logger.info("✅ 음성 재생 완료")

        except Exception as e:
            logger.error(f"❌ 음성 합성 또는 재생 오류: {e}")

    def speak_sync(self, text: str) -> None:
        """
        동기적으로 텍스트를 음성으로 변환하고 재생

        Args:
            text: 변환할 텍스트
        """
        self._synthesize_and_play(text)

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
