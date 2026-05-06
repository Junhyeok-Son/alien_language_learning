# Eridian Language Learning

> *"The Eridians don't speak. They sing."*  
> — Andy Weir, Project Hail Mary

**Project Hail Mary**에서 영감을 받은 실시간 외계 언어 번역기입니다.  
Eridian족이 사용하는 다성 화음(polyphonic chord) 시퀀스를 마이크로 입력받아,  
FAISS 벡터 검색과 Anthropic LLM을 통해 자연스러운 영어 문장으로 번역합니다.

---

## 파이프라인 개요

```
마이크 (또는 데모 신호)
       │
       ▼
  AudioCapture          sounddevice, SR=44100, chunk=2048
       │
       ▼
  AudioProcessor        롤링 버퍼(4096) → CQT 크로마(12-bin) + 스펙트로그램
       │
       ▼
  ChordHoldFilter       4 프레임 연속 감지 시 확정 (노이즈 억제)
       │
       ▼
  DynamicLexicon        FAISS IndexFlatIP — 코사인 유사도로 단어 검색
   (FAISS)              ≥0.85 고신뢰 / ≥0.70 인식 / <0.70 미지 화음
       │
       ▼
  EridianTranslator     Anthropic claude-sonnet-4-6
   (Anthropic API)      프롬프트 캐싱 + Chain-of-Thought + 대화 버퍼
       │
       ▼
  PyQt6 Dashboard       실시간 스펙트로그램, 번역 채팅, 크로마 차트
```

---

## 주요 기능

- **실시간 스펙트로그램** — PyQtGraph `ImageItem` 기반 300-컬럼 스크롤 디스플레이
- **화음 인식** — 108개 템플릿(9가지 화음 유형 × 12 음)과 코사인 유사도 매칭
- **동적 렉시콘** — FAISS 인덱스에 새 단어를 런타임에 추가 가능
- **LLM 번역** — CoT(`<think>` 태그)와 대화 컨텍스트(최근 10 교환)로 문맥 유지
- **데모 모드** — 마이크 없이 사인파 합성 신호로 전체 파이프라인 시연
- **미지 화음 학습** — 인식 불가 화음 감지 시 사용자에게 라벨링 요청

---

## 설치

### 요구 사항

- Python 3.11 이상
- Windows / macOS / Linux

### 의존성 설치

```bash
pip install -r requirements.txt
```

| 패키지 | 용도 |
|--------|------|
| `librosa` | STFT / CQT 크로마 추출 |
| `sounddevice` | 마이크 입력 스트림 |
| `numpy` / `scipy` | 신호 처리 |
| `faiss-cpu` | 벡터 유사도 검색 |
| `anthropic` | Claude API 클라이언트 |
| `PyQt6` | GUI 프레임워크 |
| `pyqtgraph` | 실시간 그래프 렌더링 |
| `python-dotenv` | 환경 변수 로드 |

### API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 Anthropic API 키를 입력합니다:

```
ANTHROPIC_API_KEY=sk-ant-...
```

> API 키가 없어도 실행 가능합니다. 이 경우 번역 결과가 `[Raw] Word1 | Word2` 형태로 출력됩니다.

---

## 실행

```bash
# 마이크 모드 (실제 오디오 입력)
python main.py

# 데모 모드 (마이크 없이 합성 신호 사용)
python main.py --demo

# 디버그 로그 출력
python main.py --demo --log-level DEBUG
```

---

## 사용 방법

| 버튼 | 기능 |
|------|------|
| `▶ Start Mic` | 마이크 입력 시작 |
| `🛸 Demo Mode` | 합성 Eridian 신호 재생 시작 |
| `■ Stop` | 오디오 처리 중단 |
| `🗑 Clear Logs` | 로그 초기화 |
| `＋ Add Word` | 화음 → 단어 수동 등록 |
| `↺ Reset Context` | LLM 대화 컨텍스트 초기화 |

화음이 4 프레임 연속 감지되면 단어로 확정되고, 4개 단어가 모이면 LLM으로 전송되어 번역 결과가 채팅 영역에 표시됩니다.

---

## 프로젝트 구조

```
alien_language_learning/
├── main.py                        # 진입점
├── requirements.txt
├── .env.example
├── data/
│   └── default_lexicon.json       # 기본 20개 화음-단어 매핑
└── src/
    ├── audio/
    │   ├── chord_templates.py     # 108개 화음 템플릿 + 매처
    │   ├── capture.py             # 마이크 입력 (sounddevice)
    │   ├── processor.py           # 크로마 추출 + 스펙트로그램
    │   └── demo_generator.py      # 합성 오디오 생성기
    ├── vector_db/
    │   └── lexicon.py             # FAISS 동적 렉시콘
    ├── llm/
    │   └── translator.py          # Anthropic 번역 엔진
    └── ui/
        ├── workers.py             # QThread 오디오/번역 워커
        └── dashboard.py           # PyQt6 메인 윈도우
```

---

## 기술 설계

### 크로마 추출 (`center=False`)

실시간 스트림에서는 미래 샘플을 사용할 수 없으므로 모든 librosa 연산에 `center=False`를 강제합니다.

```python
chroma = librosa.feature.chroma_cqt(y=audio, sr=44100, center=False)
```

### FAISS 코사인 유사도

벡터를 L2 정규화한 뒤 내적(`IndexFlatIP`)을 계산하면 코사인 유사도와 동일합니다.

| 유사도 | 판정 |
|--------|------|
| ≥ 0.85 | 고신뢰 매칭 (`✓✓`) |
| ≥ 0.70 | 일반 매칭 (`✓`) |
| < 0.70 | 미지 화음 → 사용자 라벨링 요청 |

### LLM 프롬프트 캐싱

시스템 프롬프트에 `cache_control: {"type": "ephemeral"}`을 적용하여 5분 TTL 동안 캐싱합니다.  
반복 호출 시 토큰 비용을 크게 절감할 수 있습니다.

---

## 기본 어휘 (20개)

| 단어 | 화음 | 단어 | 화음 |
|------|------|------|------|
| Friend | C major | Greet | G dom7 |
| Question | G major | Danger | A minor |
| Help | F major | Food | E minor |
| Travel | D major | Understand | G minor |
| No | C minor | Yes | A major |
| Surprise | B minor | Water | E major |
| Star | D minor | Home | F# major |
| Alive | B major | Many | G# major |
| Sun | C# major | Time | A# major |
| Energy | C dom7 | Together | F minor |

---

## 라이선스

MIT License
