# dai_AIforSecurity
사이버 공격 유형 예측 해커톤: 트래픽 속 위협을 식별하라!

# 인공지능 모델 선택
1. Task Type: 

분류(Classification), 20개의 특성을 통해 12개의 결과 중 하나로 분류 
{'GoldenEye', 'Slowloris', 'Benign', 'Web_Brute_Force', 'DDoS', 'SSH_Brute_Force', 'Botnet', 'Slow_HTTP', 'Web_XSS', 'FTP_Brute_Force', 'Port_Scanning', 'Hulk'}

1. 데이터 샘플 수: 

3000개( 소량 데이터)→ 단순모델, 선형,로지스틱 회귀

2. 특성 수, 차원:

20개 특성(네트워크 트래픽)→ attack_type 예측

- 샘플 수가 아주 적어서 n<20n<20n<20 정도라면, 비록 피처가 20개여도 p≥np\ge np≥n 상황이 돼 모델링이 어렵고 “고차원 문제”로 간주될 수 있습니다.
- 반대로 샘플이 많아서 n≫20n\gg 20n≫20 이라면 차원이 낮은 편입니다.

차원 축소(PCA)나 정규화, 피처 선택(SelectKBest, RFE) 등을 적용할 수 있는 모델인지 점검

3. 해석 가능성(Interpretability):

내부 동작 이해 가능성

4. 학습, 추론 비용:

GPU·메모리·저장공간 제약을 고려해 모델 크기와 복잡도를 결정

5. 성능요구도(정확도 vs 속도):

평가 산식 : macro f1-score: 

F1-score (클래스별): Precision과 Recall의 조화평균

Macro F1-score: 전체 C개의 클래스 각각의 F1 단순 평균낸 값

정확도>속도


# 코드 제출
- 제목에 Private 순위와 사용한 모델, 코드에 대한 keyword 기재
- 전처리, 학습, 추론 등 단계별로 코드를 작성하여 제출

# 일정
06.02
대회 시작

07.31
대회 종료

08.05
코드 및 PPT 제출 마감

08.14
코드 검증

08.18
최종 수상자 발표
