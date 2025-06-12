# LLM
MAX_TOKENS: int = 1024 * 4
TEMPERATURE: float = 0.33

# Transcribe
# https://aws.amazon.com/ko/blogs/korea/amazon-transcribe-streaming-now-supports-websockets/
# Amazon Transcribe 에서 한글은 최대 16000Hz까지 지원
SAMPLE_RATE = 16000
# 라이브 스트리밍에서는 1024 정도의 작은 청크 크기를 사용하는 것이 좋다..
CHUNK_SIZE = 512
