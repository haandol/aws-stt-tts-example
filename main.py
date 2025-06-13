import signal
import sys
import asyncio

import dotenv

dotenv.load_dotenv()  # noqa: E402

import torch
import sounddevice
import pvporcupine
import numpy as np
from silero_vad import load_silero_vad
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

from src.config import config
from src.constant import SAMPLE_RATE, CHUNK_SIZE
from src.llm import BedrockLLM
from src.tts import PollyTTS
from src.logger import logger

logger.info("🔍 환경 변수 로드 완료", config=config)

# VAD 모델 전역 변수
vad_model = None

# Porcupine wake word detector 전역 변수
porcupine = None

# 음성 입력 상태 관리
is_processing_response = False  # LLM 처리 및 TTS 재생 중인지 여부
is_wake_word_detected = False  # 웨이크워드 감지 여부

# 전역 TTS 인스턴스 (종료 신호 처리용)
tts_instance = None

# 애플리케이션 종료 플래그
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """시그널 핸들러 - Graceful shutdown"""
    logger.info(f"🛑 종료 신호 수신 (시그널: {signum})")
    shutdown_event.set()

    # 전역 TTS 인스턴스가 있다면 즉시 재생 중지
    global tts_instance
    if 'tts_instance' in globals() and tts_instance is not None:
        tts_instance.stop_playback()


# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
        audio_float = np.frombuffer(
            audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # VAD 모델 입력을 위해 torch tensor로 변환
        audio_tensor = torch.from_numpy(audio_float)

        # VAD 신뢰도 계산
        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

        # 임계값을 넘으면 음성으로 간주
        return speech_prob > threshold

    except Exception as e:
        logger.warning(f"⚠️ VAD 처리 중 오류 발생: {e}")
        return True  # 오류 시 안전하게 음성으로 간주


def initialize_porcupine():
    """Porcupine 웨이크워드 감지기 초기화"""
    global porcupine
    if porcupine is None:
        logger.info("🎯 Porcupine 웨이크워드 감지기를 초기화합니다...")
        try:
            porcupine = pvporcupine.create(
                access_key=config.porcupine_access_key,
                keywords=[config.wake_word]
            )
            logger.info("✅ Porcupine 웨이크워드 감지기가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"❌ Porcupine 초기화 실패: {e}")
            raise
    return porcupine


def detect_wake_word(audio_chunk):
    """
    오디오 청크에서 웨이크워드를 감지합니다.

    Args:
        audio_chunk: 16-bit PCM 오디오 데이터

    Returns:
        bool: 웨이크워드가 감지되면 True, 그렇지 않으면 False
    """
    try:
        # int16 PCM 데이터를 int16 배열로 변환
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

        # Porcupine으로 웨이크워드 감지
        keyword_index = porcupine.process(audio_data)

        return keyword_index >= 0
    except Exception as e:
        logger.warning(f"⚠️ 웨이크워드 감지 중 오류 발생: {e}")
        return False


async def mic_stream_with_vad(sample_rate, chunk_size):
    """VAD가 적용된 마이크 스트림"""
    global is_processing_response, is_wake_word_detected

    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    # VAD 모델 초기화
    initialize_vad()

    # Porcupine 초기화
    initialize_porcupine()

    # 음성 상태 추적 변수
    is_speaking = False
    silence_counter = 0
    silence_threshold = 30  # 약 1초간 무음이면 음성 종료로 간주 (30 * 32ms)
    wake_word_cooldown = 0  # 웨이크워드 감지 후 일정 시간 동안 재감지 방지

    def callback(indata, frame_count, time_info, status):
        if not shutdown_event.is_set():
            loop.call_soon_threadsafe(
                input_queue.put_nowait, (bytes(indata), status))

    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=sample_rate,
        callback=callback,
        blocksize=chunk_size,
        dtype="int16",
    )

    try:
        with stream:
            logger.info("🎯 VAD가 활성화된 마이크 스트리밍을 시작합니다.")

            while not shutdown_event.is_set():
                try:
                    # 타임아웃으로 종료 조건 체크
                    indata, status = await asyncio.wait_for(input_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # 응답 처리 중이면 음성 데이터 전송하지 않음
                if is_processing_response:
                    # 무음 데이터로 연결 유지
                    silence_data = bytes(len(indata))
                    yield silence_data, status
                    continue

                # 웨이크워드 감지
                if not is_wake_word_detected:
                    if wake_word_cooldown > 0:
                        wake_word_cooldown -= 1
                        yield bytes(len(indata)), status
                        continue

                    if detect_wake_word(indata):
                        logger.info("🔔 웨이크워드가 감지되었습니다!")
                        is_wake_word_detected = True
                        is_speaking = True  # 웨이크워드 감지 시 바로 음성 활동 시작
                        wake_word_cooldown = 30  # 약 1초 동안 웨이크워드 재감지 방지
                        yield indata, status  # 웨이크워드 감지 직후의 음성 데이터도 전송
                        continue
                    else:
                        # 웨이크워드가 감지되지 않았으면 무음 데이터 전송
                        yield bytes(len(indata)), status
                        continue

                # VAD로 음성 활동 감지
                has_voice = detect_voice_activity(indata, threshold=0.3)

                if has_voice:
                    if not is_speaking:
                        logger.info("🎤 음성 활동이 감지되었습니다.")
                        is_speaking = True
                    silence_counter = 0
                else:
                    if is_speaking:
                        silence_counter += 1
                        if silence_counter >= silence_threshold:
                            logger.info("🔇 음성 활동이 종료되었습니다.")
                            is_speaking = False
                            silence_counter = 0
                            is_wake_word_detected = False  # 음성 종료 시 웨이크워드 상태 초기화

                # 실제 음성 데이터 전송
                yield indata, status

    except Exception as e:
        logger.error(f"❌ 마이크 스트림 오류: {e}")
    finally:
        logger.info("🔇 마이크 스트림을 정리합니다.")
        if porcupine is not None:
            porcupine.delete()


async def write_chunks(stream):
    stream_ended = False
    try:
        async for chunk, status in mic_stream_with_vad(SAMPLE_RATE, CHUNK_SIZE):
            if shutdown_event.is_set():
                break
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
    except Exception as e:
        logger.error(f"❌ 청크 전송 오류: {e}")
    finally:
        if not stream_ended:
            try:
                await stream.input_stream.end_stream()
                stream_ended = True
                logger.info("✅ 입력 스트림이 정상적으로 종료되었습니다.")
            except Exception as e:
                if "completed" not in str(e).lower():
                    logger.warning(f"⚠️ 스트림 종료 중 오류: {e}")
                else:
                    logger.info("ℹ️ 스트림이 이미 종료되었습니다.")
                stream_ended = True

    return stream_ended


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
        global is_processing_response, is_wake_word_detected

        # 응답 처리 중이면 음성 인식 무시
        if is_processing_response:
            return

        # 웨이크워드가 감지되지 않았으면 음성 인식 무시
        if not is_wake_word_detected:
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
            is_wake_word_detected = False  # 음성 인식 종료 시 웨이크워드 상태 초기화
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
                    self.messages.append(
                        {"role": "user", "content": user_input})

                    # LLM에 전체 대화 기록 전달
                    logger.info("🤖 LLM 처리 중...")
                    response = await self.llm.model.ainvoke(self.messages)

                    # AI 응답 추가
                    ai_response = response.content
                    self.messages.append(
                        {"role": "assistant", "content": ai_response})

                    logger.info(f"🤖 AI: {ai_response}")

                    # TTS로 AI 응답을 음성으로 재생
                    logger.info("🔊 음성 재생 시작...")
                    await self.tts.speak_async(ai_response)
                    logger.info("✅ 음성 재생 완료")

                finally:
                    # 응답 처리 완료 - 음성 입력 재개
                    is_processing_response = False
                    is_wake_word_detected = False  # 응답 처리 완료 시 웨이크워드 상태 초기화
                    logger.info("▶️ 음성 입력을 재개합니다. 웨이크워드를 기다립니다.")


async def basic_transcribe(
    llm: BedrockLLM,
    tts: PollyTTS,
    sample_rate: int,
    lang_code: str,
):
    client = None
    stream = None

    try:
        client = TranscribeStreamingClient(region=config.aws_default_region)

        stream = await client.start_stream_transcription(
            language_code=lang_code,
            media_sample_rate_hz=sample_rate,
            media_encoding="pcm",
        )

        logger.info("🎙️ VAD 기반 마이크 스트리밍 채널이 열렸습니다.")
        logger.info("✨ 준비 완료! 음성으로 질문해 주세요. (AI 응답 중에는 다음 입력이 대기됩니다)")
        logger.info("💡 종료하려면 Ctrl+C를 누르세요.")

        # Instantiate our handler and start processing events
        handler = MyEventHandler(stream.output_stream, llm, tts)

                # 태스크들을 병렬로 실행하되 하나라도 완료되거나 종료 신호가 오면 정리
        write_task = asyncio.create_task(write_chunks(stream))
        handler_task = asyncio.create_task(handler.handle_events())
        shutdown_task = asyncio.create_task(wait_for_shutdown())

        tasks = [write_task, handler_task, shutdown_task]

        stream_ended = False
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # write_chunks가 완료되었다면 스트림 종료 상태 확인
            if write_task in done:
                try:
                    stream_ended = await write_task
                except Exception:
                    stream_ended = False

            # 완료되지 않은 태스크들 취소
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"❌ 태스크 실행 중 오류: {e}")

    except Exception as e:
        logger.error(f"❌ 전사 서비스 초기화 오류: {e}")
    finally:
        logger.info("🧹 리소스를 정리합니다...")

        # 스트림 정리 (아직 종료되지 않은 경우에만)
        if stream and not stream_ended:
            try:
                await stream.input_stream.end_stream()
                logger.info("✅ 입력 스트림 정리 완료")
            except Exception as e:
                if "completed" not in str(e).lower():
                    logger.warning(f"⚠️ 입력 스트림 정리 중 오류: {e}")
                else:
                    logger.info("ℹ️ 스트림이 이미 정리되었습니다.")

        # 클라이언트는 자동으로 정리되므로 명시적 close 불필요
        if client:
            logger.info("✅ 클라이언트 정리 완료")


async def wait_for_shutdown():
    """종료 신호를 대기하는 코루틴"""
    await shutdown_event.wait()
    logger.info("🛑 종료 신호를 받았습니다. 애플리케이션을 종료합니다...")


def cleanup_resources(tts=None):
    """리소스 정리 함수"""
    logger.info("🧹 최종 리소스 정리를 시작합니다...")

    try:
        # TTS 리소스 정리
        if tts is not None:
            tts.cleanup()
    except Exception as e:
        logger.warning(f"⚠️ TTS 정리 중 오류: {e}")

    try:
        # sounddevice 정리
        sounddevice.stop()
        logger.info("✅ sounddevice 정리 완료")
    except Exception as e:
        logger.warning(f"⚠️ sounddevice 정리 중 오류: {e}")

    try:
        # torch 스레드 정리
        if vad_model is not None:
            torch.set_num_threads(1)
        logger.info("✅ torch 리소스 정리 완료")
    except Exception as e:
        logger.warning(f"⚠️ torch 정리 중 오류: {e}")

    logger.info("✅ 최종 리소스 정리 완료")


if __name__ == "__main__":
    tts = None
    try:
        # VAD 초기화
        initialize_vad()
        logger.info("✅ Silero VAD가 활성화되었습니다.")

        # LLM 초기화
        llm = BedrockLLM(
            model_id=config.model_id,
            aws_profile_name=config.aws_profile,
        )

        # TTS 초기화
        tts = PollyTTS(
            voice_id=config.voice_id,
            aws_profile=config.aws_profile,
        )

        # 전역 TTS 인스턴스 설정 (종료 신호 처리용)
        tts_instance = tts

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
        except KeyboardInterrupt:
            logger.info("🛑 사용자에 의해 중단되었습니다.")
        except Exception as e:
            logger.error(f"❌ 실행 중 오류 발생: {e}")
        finally:
            logger.info("🏁 이벤트 루프를 정리합니다...")

            # 남은 태스크들을 정리
            pending_tasks = asyncio.all_tasks(loop)
            if pending_tasks:
                logger.info(f"⚠️ {len(pending_tasks)}개의 미완료 태스크를 취소합니다...")
                for task in pending_tasks:
                    task.cancel()

                # 모든 태스크가 정리될 때까지 대기
                try:
                    loop.run_until_complete(
                        asyncio.gather(*pending_tasks, return_exceptions=True)
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 태스크 정리 중 오류: {e}")

            # 이벤트 루프 정리
            try:
                loop.close()
                logger.info("✅ 이벤트 루프가 정리되었습니다.")
            except Exception as e:
                logger.warning(f"⚠️ 이벤트 루프 정리 중 오류: {e}")

    except Exception as e:
        logger.error(f"❌ 애플리케이션 초기화 오류: {e}")
    finally:
        # 최종 리소스 정리
        cleanup_resources(tts)
        logger.info("👋 애플리케이션이 완전히 종료되었습니다.")
        sys.exit(0)
