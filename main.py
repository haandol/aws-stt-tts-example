import asyncio

import dotenv

dotenv.load_dotenv()  # noqa: E402

import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from src.config import config
from src.constant import SAMPLE_RATE, CHUNK_SIZE
from src.llm import BedrockLLM
from src.logger import logger

logger.info("🔍 환경 변수 로드 완료", config=config)


class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream: TranscriptResultStream, llm: BedrockLLM):
        super().__init__(transcript_result_stream)
        self.llm = llm
        self.is_listening = False
        self.messages = [
            {
                "role": "system",
                "content": "당신은 유용한 AI 비서입니다. 모든 답변은 사용자의 언어로 답변해 주세요.",
            }
        ]

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
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

                # 사용자 메시지 추가
                self.messages.append({"role": "user", "content": user_input})

                # LLM에 전체 대화 기록 전달
                response = await self.llm.model.ainvoke(self.messages)

                # AI 응답 추가
                ai_response = response.content
                self.messages.append({"role": "assistant", "content": ai_response})

                logger.info(f"🤖 AI: {ai_response}")


async def mic_stream(sample_rate, chunk_size):
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=sample_rate,
        callback=callback,
        blocksize=chunk_size,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    async for chunk, status in mic_stream(SAMPLE_RATE, CHUNK_SIZE):
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def basic_transcribe(
    llm: BedrockLLM,
    sample_rate: int,
    lang_code: str,
):
    client = TranscribeStreamingClient(region=config.aws_default_region)

    stream = await client.start_stream_transcription(
        language_code=lang_code,
        media_sample_rate_hz=sample_rate,
        media_encoding="pcm",
    )

    logger.info("🎙️ 마이크 스트리밍 채널이 열렸습니다. 말씀해 주세요.")

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream, llm)
    await asyncio.gather(write_chunks(stream), handler.handle_events())


if __name__ == "__main__":
    # TODO: VAD 추가

    # LLM 초기화
    llm = BedrockLLM(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        aws_profile_name=config.aws_profile,
    )

    # 새 이벤트 루프 명시적으로 설정
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            basic_transcribe(
                llm=llm,
                sample_rate=SAMPLE_RATE,
                lang_code=config.lang_code,
            )
        )
    finally:
        loop.close()
