# KMMLU Criminal-Law RAG Agent

이 프로젝트는 KMMLU Criminal-Law 벤치마크(HAERAE-HUB/KMMLU: “Criminal-Law” test split)를  
RAG(Retrieval-Augmented Generation) 방식으로 평가하기 위한 파이프라인입니다.  
“형법”·“형사소송법” 등 공식 법령 조문을 인덱싱하여 FAISS 기반 Retriever를 구축하고,  
GPT-4o-mini + OpenAI Batch API로 성능을 측정합니다.

## Project structure

```text
.
├── Dockerfile
├── docker-compose.yaml
├── run_all.sh
├── pyproject.toml
├── poetry.lock
├── README.md
├── .env                        # 환경 변수(OPENAI_API_KEY, LAW_SERVICE_OC)
├── scripts/
│   ├── load_data.py            # 국가법령정보 API 검색, 저장 및 JSONL 변환
│   ├── prepare_data.py         # Raw data로부터 embedding 생성
│   ├── build_batch_input.py    # 평가용 batch input JSONL 생성
│   └── evaluate.py             # KMMLU criminal-law 테스트셋 평가
├── data/
│   ├── raw/
│   │   ├── law_articles.jsonl  # load_data.py 출력
│   ├── batch/                  # batch input / output JSONL
│   ├── embeddings/             # FAISS index 파일
│   └── score.txt               # KMMLU 평가 결과
└── agent/
    ├── retriever.py            # FAISS 기반 Retriever
    └── llm.py                  # GPT-4o-mini RAG Agent
```

## Get started

1. Git clone

    레포지토리에 접근을 원하시면 <inseonyu7@gmail.com>으로 연락 바랍니다.

    ```bash
    git clone https://github.com/Insun35/KMMLU-criminal-law-test
    cd KMMLU-criminal-law-test
    ```

2. Set `.env`

    프로젝트 루트에 `.env`를 생성하고 OpenAI API 키와 국가법령정보 API 요청이 승인된 아이디를 넣습니다.
    `LAW_SERVICE_OC`는 등록된 이메일의 앞부분입니다. (예: <alice123@gmail.com> -> `LAW_SERVICE_OC=alice123`)

    ```
    OPENAI_API_KEY=sk-…
    LAW_SERVICE_OC=your_law_api_oc
    ```

### Run on local

```bash
# poetry 설치
pip install poetry

# 전체 스크립트 실행 권한 부여
chmod +x run_all.sh

# 전체 스크립트 실행
run_all.sh

# 또는 스크립트 순차 실행
poetry run python -m scripts.load_data

poetry run python -m scripts.prepare_data

poetry run python -m scripts.build_batch_input

poetry run python -m scripts.evaluate
```

### Run on Docker

```bash
# Docker 빌드 및 실행
docker build .
docuer run -p [Local Port]:[Process Port in Container] [ Image ID ]
```

## Details

- FAISS Retriever
  - agent/retriever.py 에서 /data/embeddings/lbox_index.faiss 로드
  - text-embedding-3-small 로 query embedding → IP 검색 → top-k 청크 반환

- Agent
  - agent/agent.py 에서 Retriever + GPT-4o-mini 호출
  - Prompt 템플릿에 “법령명 + 제N조 제목 + 본문” 컨텍스트 포함

- Batch API 평가
  - build_batch_input.py → /openai_batch/eval_input.jsonl
  - evaluate.py → client.files.create + client.batches.create → 상태 폴링
  - eval_output.jsonl → 정확도 계산 → score.txt
