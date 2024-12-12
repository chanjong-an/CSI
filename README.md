건설안전 사고 예측 모델

프로젝트 개요

본 프로젝트는 건설안전 사고 데이터를 기반으로 사고 유형을 예측하는 모델을 구축하고, 사고 데이터를 효과적으로 분류 및 분석하는 데 목적이 있습니다. 이를 통해 건설 현장에서 발생할 수 있는 사고를 사전에 예측하고 예방 조치를 제안할 수 있는 기반을 마련합니다.

사용된 데이터는 국토안전관리원_건설안전사고사례 데이터셋이며, 주요 작업은 데이터 전처리, 머신러닝 모델 학습 및 평가로 구성됩니다.

구현 기능
	1.	데이터 전처리:
	•	텍스트 데이터의 특수문자 제거 및 형태소 분석.
	•	사고 유형의 단순화 및 불필요한 데이터 제거.
	•	TF-IDF 벡터화를 통한 텍스트 특성 추출.
	2.	모델 학습 및 예측:
	•	XGBoost를 활용한 사고 유형 예측 모델 학습.
	•	교차 검증 및 테스트 데이터 평가를 통한 성능 검증.
	•	두 번째 예측 결과를 기반으로 모델 재학습 및 파라미터 조정.
	3.	결과 분석:
	•	검증 데이터로 최종 모델의 성능 평가.
	•	첫 번째 및 두 번째 예측 결과의 정확도 비교.

전처리 과정

1. 데이터 정제
	•	결측치 제거: 데이터셋 내 결측값을 제거하여 분석의 정확성을 높였습니다.
	•	특수문자 제거: 텍스트 데이터에서 불필요한 특수문자를 제거하여 텍스트 일관성을 확보했습니다.

X = df['사고경위'].apply(lambda x: re.sub(r'[-=+,#/\?:^$.@*"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', ' ', x))



2. 형태소 분석
	•	**한국어 형태소 분석기(Okt)**를 사용하여 텍스트를 분석하고, 주요 단어(명사와 동사)를 추출했습니다.

def custom_tokenizer(text):
    okt = Okt()
    tokens = okt.pos(text, stem=True)
    return [token[0] for token in tokens if token[1] in ("Noun", "Verb")]



3. 사고 유형 단순화
	•	복잡한 사고 유형을 주요 카테고리로 통합하여 분석 효율성을 높였습니다.

df['인적사고'] = df['인적사고종류'].replace({
    '넘어짐(기타)': '넘어짐',
    '떨어짐(2미터 미만)': '떨어짐',
    ...
})



4. 데이터 분리
	•	훈련(80%), 테스트(10%), 검증(10%) 데이터로 분리하여 모델 학습과 성능 평가를 수행했습니다.

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)



5. TF-IDF 벡터화
	•	텍스트 데이터를 수치화하기 위해 TF-IDF(Vectorization)를 적용했습니다.

tfidf_vect = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=2, max_df=2000)
X_train_tf = tfidf_vect.fit_transform(X_train)

학습 및 재학습 과정

1. 초기 모델 학습
	•	XGBoost를 사용하여 첫 번째 예측을 수행.
	•	첫 번째 예측 결과와 실제 값이 일치하지 않을 경우, 두 번째 예측 결과를 확인.
	•	1차 정확도: 0.721
    •	두 번째 예측이 실제와 일치한 비율: 0.144

2. 파라미터 조정 및 재학습
	•	두 번째 예측 결과를 반영하여 모델 학습률(learning_rate)과 최대 깊이(max_depth)를 동적으로 조정.

params["learning_rate"] = max(0.01, params["learning_rate"] * 0.8)
params["max_depth"] = min(10, params["max_depth"] + 1)


	•	조정된 파라미터:

{'objective': 'multi:softprob', 'num_class': 7, 'random_state': 42, 'tree_method': 'hist', 'learning_rate': 0.06400000000000002, 'max_depth': 8}



3. 최종 검증
	•	검증 데이터를 사용하여 최종 모델 성능 평가.
	•	검증 데이터 최종 정확도: 0.729

향후 개선 방향
	1.	모델 성능 개선:
	•	추가적인 하이퍼파라미터 튜닝.
	•	더 많은 데이터로 학습하여 모델 일반화 성능 향상.
	2.	추가 기능:
	•	예측 결과를 시각화하여 직관적 인사이트 제공.
	•	사고 예방을 위한 추천 시스템 개발.

본 프로젝트는 텍스트 기반 사고 데이터 분석과 예측 모델 학습을 통해, 건설 현장에서의 안전 관리 향상을 목표로 하고 있습니다.
