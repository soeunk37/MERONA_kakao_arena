# MERONA_kakao_arena
카카오아레나 멜론 플레이리스트 추천

## 1. 노래
- DAE를 이용한 협업 필터링
- best model & 데이터 set: https://drive.google.com/drive/folders/1GzXRGccUHFvuVvRnYB_rGt2VmRskajeO?usp=sharing
- 결과
  - **nDCG@3000** : 0.00521
  - **Recall@3000** : 0.01782

## 2. 태그
- Word2Vec을 활용한 제목 유사도 산출
- 데이터 전처리(Word2Vec을 활용한 태그 수 줄이기)
- 전처리 된 데이터 set: https://drive.google.com/file/d/1W4YxUvBYxfTeV9kgO46AwZleaod0Hmdi/view?usp=sharing
- 결과
  - **Recall** : 0.05491
