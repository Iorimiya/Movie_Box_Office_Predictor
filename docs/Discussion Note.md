# Discussion Note

## before 2025/08/03

#### TODO

##### 使用新OO架構重構程式

- [X] 

##### SKEP

- [x] 測試環境相容性
- [ ] 測試程式相容性
	- [x] 新的環境和SKEP本身的相容性
	- [ ] 新環境和原本的scraping code、prediction model的相容性
		- [ ] class測試
			- [ ] function測試
- [ ] 測試pretrained model效能
- [ ] 遷入主專案
- [ ] 訓練新prediciton model
- [ ] 評估prediction model效能

##### BERT

- [ ] 測試環境相容性
- [ ] 遷入主專案
- [ ] 測試程式相容性
- [ ] 建立符合BERT訓練輸入的資料集
- [ ] 訓練新prediciton model
- [ ] 評估prediction model效能


#### Waiting for Discussion

##### Method to optimize model

- change learning rate
- change limit week
- change layers parameter
- using MovieSessionData
- using MovieSessionData with One Movie One Data
- GRU replace LSTM
- Stop training when f1 not increase
- 可變長度序列嵌入到固定長度特徵向量(新研究)
- OverSampling、UnderSampling

## 2025/08/03

#### TODO

- [x] Continue Refactoring Prediction Code
  - [x] finish prediction data processor
  - [x] finish prediction model core
  - [x] finish prediction pipeline
  - [x] finish prediction evaluator
  - [x] fix prediction data preprocess bug
  - [x] fix prediction duplicate scaling bug
  - [x] change prediction continuous data selection
  - [x] Sentiment Model f1-score
- [ ] Test Sentiment Model with ChnSentiCorp dataset
  - [x] environment 衝突
- [ ] ELECTRA environment
- [ ] change base of analyse in SKEP
  - Positive -> Positive
  - Negative with score upper than 0.8 -> Negative
  - Negative with score lower or equal than 0.8 -> Positive

#### Waiting for Discussion


- Data Collection (Sentiment)
- Feature Engineering (Sentiment)
  - 遇到的問題：
  - 學習到的不是資料集本身，而是製造句子時使用的其他部分「非常推薦、完全不推薦」
  - 目前改用「不製造句子」的方式
  - model1_e1
    - 問題
    - 只是一個分類器，只對資料集中的單詞有反應，不是的話->亂猜
    - 可能的解決方法
      - lexicon_sentiment_data_v2.csv
      - 內部的word都是一整段句子
- Training (Sentiment)
- Evaluation (Sentiment)
- Deployment (Sentiment)
- Data Collection (Prediction)
- Feature Engineering (Prediction)
- Training (Prediction)
- Evaluation (Prediction)


- [X] 修改Sentiment Model
- [X] Sentiment Model在Chnsenticorp的評估
- [ ] 比較三種模型
- [ ] GPT的比較
- [ ] 第二次電影不採納
