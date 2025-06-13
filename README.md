# AWS STT-TTS 예제 🎤🤖🔊

AWS의 음성 인식(Transcribe), 대화형 AI(Bedrock), 음성 합성(Polly) 서비스를 활용한 실시간 음성 대화 시스템입니다.

## 📋 기능

- **웨이크워드 감지**: Picovoice Porcupine을 사용한 음성 활성화
- **실시간 음성 인식**: AWS Transcribe를 통한 실시간 음성-텍스트 변환
- **음성 활동 감지(VAD)**: Silero VAD 모델을 사용한 지능적 음성 구간 감지
- **AI 대화**: AWS Bedrock의 Claude 모델을 통한 자연스러운 대화
- **음성 합성**: AWS Polly를 통한 AI 응답의 음성 변환
- **연속 대화**: 대화 기록을 유지하며 맥락을 이해하는 대화

## 🛠️ 기술 스택

- **Python 3.13+**
- **AWS 서비스**:
  - AWS Transcribe (음성 인식)
  - AWS Bedrock (Claude LLM)
  - AWS Polly (음성 합성)
- **음성 처리**:
  - Picovoice Porcupine (웨이크워드 감지)
  - Silero VAD (음성 활동 감지)
  - SoundDevice (마이크 입력)
  - PyTorch (VAD 모델)

## 📦 설치

### 1. 프로젝트 클론

```bash
git clone <repository-url>
cd aws-stt-tts-example
```

### 2. Python 환경 설정

이 프로젝트는 Python 3.13 이상이 필요합니다.

```bash
uv sync

```

### 3. AWS 자격 증명 설정

AWS CLI를 통해 자격 증명을 설정하거나, IAM 역할을 사용하세요.

```bash
aws configure
```

필요한 AWS 권한:
- `transcribe:StartStreamTranscription`
- `bedrock:InvokeModel`
- `polly:SynthesizeSpeech`

## ⚙️ 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```env
# AWS 설정
AWS_DEFAULT_REGION=us-east-1
AWS_PROFILE=default

# AI 모델 설정
MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0

# 음성 인식 언어 설정
LANG_CODE=ko-KR

# TTS 음성 설정 (한국어 음성)
VOICE_ID=Jihye

# 환경 구분
ENVIRONMENT=local

# Porcupine 웨이크워드 설정
PORCUPINE_ACCESS_KEY=your_access_key_here
PORCUPINE_WAKE_WORD=computer
```

### 환경 변수 설명

- `AWS_DEFAULT_REGION`: AWS 서비스를 사용할 리전
- `AWS_PROFILE`: 사용할 AWS 프로필 (선택사항)
- `MODEL_ID`: Bedrock에서 사용할 AI 모델 ID
- `LANG_CODE`: 음성 인식 언어 코드 (`ko-KR`, `en-US` 등)
- `VOICE_ID`: Polly TTS 음성 ID (`Seoyeon`, `Jihye` 등)
- `ENVIRONMENT`: 실행 환경 구분
- `PORCUPINE_ACCESS_KEY`: Picovoice Porcupine 웨이크워드 서비스 접근 키
- `PORCUPINE_WAKE_WORD`: 웨이크워드 단어

## 🚀 사용법

### 애플리케이션 실행

```bash
# 의존성 설치
uv sync

# 애플리케이션 실행
uv run python main.py
```

### 사용 방법

1. 애플리케이션을 실행하면 마이크가 활성화됩니다
2. "computer" 웨이크워드를 말하여 시스템을 활성화하세요
3. 웨이크워드 감지 후 질문이나 대화를 시작하세요
4. AI가 음성으로 응답합니다
5. 응답이 끝나면 다시 웨이크워드를 기다립니다

### 주요 특징

- **웨이크워드 활성화**: "computer" 웨이크워드로 시스템 활성화
- **지능적 음성 감지**: VAD를 통해 음성 시작/종료를 자동으로 감지
- **응답 중 입력 차단**: AI가 응답하는 동안 새로운 음성 입력을 일시 정지
- **대화 맥락 유지**: 이전 대화 내용을 기억하여 자연스러운 대화 진행

## 📁 프로젝트 구조

```
aws-stt-tts-example/
├── main.py                 # 메인 애플리케이션
├── src/
│   ├── config.py          # 환경 설정 관리
│   ├── constant.py        # 상수 정의
│   ├── llm.py            # Bedrock LLM 래퍼
│   ├── tts.py            # Polly TTS 래퍼
│   └── logger.py         # 로깅 설정
├── pyproject.toml        # 프로젝트 설정 및 의존성
├── .env                  # 환경 변수 (생성 필요)
└── README.md
```

### 의존성 관리

이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다:

```bash
uv sync
```