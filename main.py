import asyncio
import numpy as np
import torch

import dotenv

dotenv.load_dotenv()  # noqa: E402

import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

from silero_vad import load_silero_vad
from src.config import config
from src.constant import SAMPLE_RATE, CHUNK_SIZE
from src.llm import BedrockLLM
from src.tts import PollyTTS
from src.logger import logger

logger.info("🔍 환경 변수 로드 완료", config=config)

# VAD 모델 전역 변수
vad_model = None

# 음성 입력 상태 관리
is_processing_response = False  # LLM 처리 및 TTS 재생 중인지 여부


def initialize_vad():
    """VAD 모델 초기화"""
    global vad_model
    if vad_model is None:
        logger.info("🎯 Silero VAD 모델을 초기화합니다...")
        vad_model = load_silero_vad()
        torch.set_num_threads(1)
        logger.info("✅ Silero VAD 모델이 초기화되었습니다.")
    return vad_model


def detect_voice_activity(audio_chunk, threshold=0.5):
    """
    오디오 청크에서 음성 활동을 감지합니다.

    Args:
        audio_chunk: 16-bit PCM 오디오 데이터
        threshold: VAD 신뢰도 임계값 (0.0-1.0)

    Returns:
        bool: 음성이 감지되면 True, 그렇지 않으면 False
    """
    try:
        # int16 PCM 데이터를 float32로 변환 (-1.0 ~ 1.0 범위)
        audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # VAD 모델 입력을 위해 torch tensor로 변환
        audio_tensor = torch.from_numpy(audio_float)

        # VAD 신뢰도 계산
        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

        # 임계값을 넘으면 음성으로 간주
        return speech_prob > threshold

    except Exception as e:
        logger.warning(f"⚠️ VAD 처리 중 오류 발생: {e}")
        return True  # 오류 시 안전하게 음성으로 간주


class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream: TranscriptResultStream, llm: BedrockLLM, tts: PollyTTS):
        super().__init__(transcript_result_stream)
        self.llm = llm
        self.tts = tts
        self.is_listening = False
        self.voice_buffer = []  # 음성 버퍼
        self.silence_counter = 0  # 무음 카운터
        self.silence_threshold = 30  # 무음 임계값 (약 1초 = 30 * 32ms)
        self.min_speech_chunks = 5  # 최소 음성 청크 수
        self.messages = [
            {
                "role": "system",
                "content": "당신은 유용한 AI 비서입니다. 모든 답변은 사용자의 언어로 답변해 주세요.",
            }
        ]

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        global is_processing_response

        # 응답 처리 중이면 음성 인식 무시
        if is_processing_response:
            return

        results = transcript_event.transcript.results

        # 음성 인식 시작 감지
        if results and not self.is_listening:
            logger.info("🎤 음성 인식이 시작되었습니다.")
            self.is_listening = True

        # 음성 인식 종료 감지
        if not results and self.is_listening:
            logger.info("🔇 음성 인식이 종료되었습니다.")
            self.is_listening = False
            return

        for result in results:
            # 음성 인식 종료 감지 (결과가 최종인 경우)
            if not result.is_partial and self.is_listening:
                self.is_listening = False
                logger.info("🔇 음성 인식이 종료되었습니다.")

                user_input = result.alternatives[0].transcript
                logger.info(f"🔇 음성 인식 결과: {user_input}")

                # 응답 처리 시작 - 다음 음성 입력 차단
                is_processing_response = True
                logger.info("⏸️ 음성 입력을 일시 정지합니다.")

                try:
                    # 사용자 메시지 추가
                    self.messages.append({"role": "user", "content": user_input})

                    # LLM에 전체 대화 기록 전달
                    logger.info("🤖 LLM 처리 중...")
                    response = await self.llm.model.ainvoke(self.messages)

                    # AI 응답 추가
                    ai_response = response.content
                    self.messages.append({"role": "assistant", "content": ai_response})

                    logger.info(f"🤖 AI: {ai_response}")

                    # TTS로 AI 응답을 음성으로 재생 (지혜 목소리 사용)
                    logger.info("🔊 음성 재생 시작...")
                    await self.tts.speak_async(ai_response, voice_id=config.voice_id)
                    logger.info("✅ 음성 재생 완료")

                finally:
                    # 응답 처리 완료 - 음성 입력 재개
                    is_processing_response = False
                    logger.info("▶️ 음성 입력을 재개합니다. 다시 말씀해 주세요.")


async def mic_stream_with_vad(sample_rate, chunk_size):
    """VAD가 적용된 마이크 스트림"""
    global is_processing_response

    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    # VAD 모델 초기화
    initialize_vad()

    # 음성 상태 추적 변수
    is_speaking = False
    silence_counter = 0
    silence_threshold = 30  # 약 1초간 무음이면 음성 종료로 간주 (30 * 32ms)

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=sample_rate,
        callback=callback,
        blocksize=chunk_size,
        dtype="int16",
    )

    with stream:
        logger.info("🎯 VAD가 활성화된 마이크 스트리밍을 시작합니다.")

        while True:
            indata, status = await input_queue.get()

            # 응답 처리 중이면 음성 데이터 전송하지 않음 (단, 무음 데이터는 전송하여 연결 유지)
            if is_processing_response:
                # 무음 데이터로 연결 유지
                silence_data = bytes(len(indata))  # 무음 데이터 생성
                yield silence_data, status
                continue

            # VAD로 음성 활동 감지 (낮은 임계값으로 설정하여 더 민감하게)
            has_voice = detect_voice_activity(indata, threshold=0.3)

            if has_voice:
                if not is_speaking:
                    logger.info("🎤 음성 활동이 감지되었습니다.")
                    is_speaking = True
                silence_counter = 0
            else:
                if is_speaking:
                    silence_counter += 1
                    # 짧은 무음은 허용 (말하는 중간의 짧은 멈춤)
                    if silence_counter >= silence_threshold:
                        logger.info("🔇 음성 활동이 종료되었습니다.")
                        is_speaking = False
                        silence_counter = 0

            # 실제 음성 데이터 전송
            yield indata, status


async def write_chunks(stream):
    async for chunk, status in mic_stream_with_vad(SAMPLE_RATE, CHUNK_SIZE):
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def basic_transcribe(
    llm: BedrockLLM,
    tts: PollyTTS,
    sample_rate: int,
    lang_code: str,
):
    client = TranscribeStreamingClient(region=config.aws_default_region)

    stream = await client.start_stream_transcription(
        language_code=lang_code,
        media_sample_rate_hz=sample_rate,
        media_encoding="pcm",
    )

    logger.info("🎙️ VAD 기반 마이크 스트리밍 채널이 열렸습니다.")
    logger.info("✨ 준비 완료! 음성으로 질문해 주세요. (AI 응답 중에는 다음 입력이 대기됩니다)")

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream, llm, tts)
    await asyncio.gather(write_chunks(stream), handler.handle_events())


if __name__ == "__main__":
    # VAD 초기화
    initialize_vad()
    logger.info("✅ Silero VAD가 활성화되었습니다.")

    # LLM 초기화
    llm = BedrockLLM(
        model_id=config.model_id,
        aws_profile_name=config.aws_profile,
    )

    # TTS 초기화 (지혜 목소리 사용)
    tts = PollyTTS(aws_profile_name=config.aws_profile)

    # 새 이벤트 루프 명시적으로 설정
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            basic_transcribe(
                llm=llm,
                tts=tts,
                sample_rate=SAMPLE_RATE,
                lang_code=config.lang_code,
            )
        )
    finally:
        loop.close()
