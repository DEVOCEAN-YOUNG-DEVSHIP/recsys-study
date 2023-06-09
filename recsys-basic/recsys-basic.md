# 추천시스템(Recommender System)

추천시스템이란 사용자(User)가 선호하는 아이템(Item)을 예측하는 시스템입니다.\
\
넷플릭스, 멜론, 각종 쇼핑몰 등 거의 모든 분야에서 추천시스템이 쓰이고 있습니다. 
사용자의 관심분야의 정보들만 추려주는 추천시스템은 정보의 홍수 속에서 사용자들의 피로도를 줄여줍니다.\
\
이번 포스팅에서는 추천시스템에 대한 간단한 설명과 기본적인 추천시스템 알고리즘에 대해서 알아보려고 합니다 :)      


# 추천시스템의 목표

#### **1) Relevance**
- 추천시스템의 명백한 목적
- 추천된 아이템이 유저와 관련이 있는가?

#### **2) Novelty**
- 보편적인 아이템(ex.Top100)이 아닌 유저가 탐색하지 못했던 관심분야의 새로운 아이템인가?

#### **3) Serendipity**
- 유저가 이전에 경험해보지 못한 완전한 새로운 아이템인가?
- 추천시스템의 부작용인 필터버블을 방지하는 중요한 요소

위 세가지의 목표를 적절하게 조화한 것이 좋은 추천시스템이라고 할 수 있습니다.


# 추천시스템의 기본 알고리즘
![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/e8258529-9e21-4c43-98f1-01b0f8534909)

### 1. 콘텐츠 기반 필터링(Content based filtering)

사용자들은 한 장르의 음악을 많이 듣거나 특정 브랜드의 상품을 자주 구매하는 등 기존에 소비한 아이템과 유사한 아이템을 소비하는 경우가 많습니다.\
이와 같이 소비 패턴이 뚜렷하다면 아이템의 특징을 활용하여 추천하는 콘텐츠 기반 필터링 방식이 효과적입니다.\
\
콘텐츠 기반 필터링은 사용자가 소비한 아이템에 대해 해당 아이템과 내용(content)이 비슷하거나 특별한 관계가 있는 다른 아이템을 추천하는 방식입니다. 이때, 아이템의 내용(Content)은 이름이나 장르같은 텍스트 데이터, 이미지 데이터 등으로 나타냅니다.

![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/25badad5-f939-4abb-a9c6-548d8f31dbe2)

\
아이템 간의 유사도를 구하는 방법은 다음과 같습니다.\
먼저 아이템을 벡터 형태로 표현한 후 벡터 간의 유사도를 계산합니다.\
\
**(1) 아이템을 벡터형태로 표현하기** \
아이템을 벡터형태로 표현하는 방법에는 One-hot encoding과 Embedding 방법이 있습니다.
    One-hot encoding은 아이템의 카테고리와 같은 범주형 데이터를 표현하는 방법으로 표현해야하는 범주의 개수를 크기로 갖는 벡터를 만들어 데이터를 0과 1로 표현하는 방법입니다.\
    ![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/e305e8cd-b35b-440b-ad7f-6bb3d3ef5f6e)

그러나 표현해야하는 범주형 데이터 개수가 많아지면(ex.이미지, 장문의 글) 벡터의 크기가 너무 커지므로 고정된 크기의 벡터로 데이터를 표현하는 Embedding 방법을 사용합니다.\
    \
텍스트 데이터의 경우 주로 Word2Vec 모델로 각 단어의 임베딩 벡터를 학습하고 텍스트에 등장하는 단어의 벡터를 합하거나 TF-IDF 가중 합산을 하는 방식을 사용합니다. 또한, 문맥을 고려하기 위한 CNN, RNN 모델들이 제안되었으면 최근에는 대규모 언어모델인 BERT로 임베딩을 하는 연구도 진행되고 있습니다.\
    \
이미지 데이터의 경우에는 pretrained된 이미지 분류 모델을 가져와서 분류 레이어(classification layer)에 입력값으로 들어가는 Bottleneck feature을 이미지 임베딩 벡터로 사용합니다. 즉, softmax 함수에 들어가서 분류되기 전 과정에서 생성되어 있는 이미지 특징 벡터를 사용합니다.

**(2) 벡터 간 유사도 구하기** \
위 과정에서 추출한 벡터들 간에 유사도를 구하는 방법에는 내적, 코사인 유사도, 피어슨 상관 계수 등이 있습니다.\
\
최종적으로 구한 유사도를 나열하여 높은 유사도를 보인 아이템을 추천해주면 컨텐츠 기반 추천시스템은 완료됩니다.\
\
\
콘텐츠 기반 필터링은 아이템 정보만으로 추천이 가능하기 때문에 콜드 스타트 문제가 발생하지 않습니다. 그러나 아이템 속성에만 의존하기 때문에 협업 필터링에 비해 추천 성능이 떨어지기 때문에 주로 추천 대상 아이템이 빠르게 바뀌거나 소비이력이 적은 아이템에 대해 협업 필터링을 보완하는 용도로 많이 활용됩니다.

___
### 2. 협업 필터링(Collaborative filtering)

협업 필터링이란 사용자의 행동패턴을 이용해 추천하는 시스템입니다. 한 사용자와 비슷한 행동패턴을 보인 사용자들의 아이템을 추천해주는 원리입니다. 사용자와 비슷한 취향의 사람들이 좋아하는 것은 사용자도 좋아할 가능성이 높다는 가정을 전제로 합니다. 따라서 사용자 기반 필터링이라고도 합니다.


![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/db1d4f53-cc2c-4509-8133-256b0e8d458b)
\
협업 필터링에는 Memory-based CF와 Model-based CF가 있습니다.

**(1) Memory-based Collaborative Filtering** 

**1) User-based Filtering**
: "평점 유사도"를 기반으로 나와 유사한 사용자를 찾은 후, 그 유사한 사용자가 좋아한 아이템을 추천합니다.
- 특정 사용자를 기준
ex) 나와 비슷한 사용자 "ㄱ"은 "B"노래도 좋아했습니다.
![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/33b1eb8a-d4a0-420e-86c6-78798cdfb3f0)

위 그림에서 5번째 사용자에게 영화를 추천해준다고 했을 때, 가장 유사도가 높은 3번째 사용자이므로 해당 사용자가 좋아한 영화를 추천해주게 됩니다.

**2) Item-based Filtering**
: 특정 아이템을 좋아한 사용자들을 찾은 후, 그 사용자들이 공통적으로 좋아했던 다른 아이템을 찾아서 추천합니다.
- 특정 아이템 기준
ex) "A"노래를 좋아한 사용자는 "B"노래도 좋아했습니다.
![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/eb360085-d94b-4827-9df0-a5bfcb0965e2)

\
\
사용자 간 유사도를 구하는 구체적인 계산 방법은 아래 사이트를 참고하시기 바랍니다.\
https://medium.com/@toprak.mhmt/collaborative-filtering-3ceb89080ade \
위 사이트를 정리해 놓은 블로그\
https://kmhana.tistory.com/31?category=882777



그러나 Memory-based CF는 데이터가 축적되지 않았거나 sparse한 경우 성능이 낮으며 데이터가 너무 많아지면 속도가 저하되는 단점이 있습니다. 따라서 Model-based CF로 이를 보완합니다.\
\
**(2) Model-based Collaborative Filtering** \
머신러닝 알고리즘을 통해 사용자가 아직 평가하지 않은 아이템의 평점을 예측하는 등 모델이 데이터를 학습하여 데이터 정보를 압축합니다.\
항목 간 유사성 계산에서 벗어나 데이터의 패턴을 학습합니다. 즉, A와 B의 유사도를 구하는 것이 아닌 A, B 각각에 대한 특징을 요약하는 방식입니다.\
Model-based 알고리즘 종류는 다음과 같습니다.
![image](https://github.com/DEVOCEAN-YOUNG-DEVSHIP/recsys-study/assets/98035735/545b9fde-066a-4af3-ba7b-59ae29449be3)

모델 기반 협업 필터링의 장점은
1. 대용량의 데이터를 행렬로 구성하여 압축된 작은 형태로 저장할 수 있습니다.
2. 미리 학습된 모델로 추천해주는 속도가 빠릅니다.
3. 다양한 범위의 추천을 할 수 있습니다.
   메모리 기반 협업필터링과 다르게 없는 데이터를 예측해서 학습하기 때문에 평점이 매겨지지 않은 아이템들에 대해서도 고려하고 주어진 데이터보다 더 넓게 추천할 수 있습니다.
