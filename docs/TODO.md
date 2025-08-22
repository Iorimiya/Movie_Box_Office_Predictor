- [X] Sentiment評估用資料集(for Sentiment Model, SKEP, ELECTRA)
  - [X] ChnSentiCorp dataset
- [X] Refactor main code
  - [X] model
  - [X] PR
- [ ] ELECTRA environment
- [ ] 比較數據集對各Model的效能
- [ ] Prediction 的 Feature 調整
  - 本週讚總數
  - 本週噓總數
  - 計算每一篇Review的讚和噓的比例，加以平均
  - 計算本週所有讚和噓的比例
- [ ] 增加評論資料中讚和虛的數量
  - [ ] 增加PublicReview的子類別PttReview
  - [ ] 在review_collector中增加蒐集的部分
  - [ ] 在WeekData中增加Feature想呈現的資料
  - [ ] 在DataProcessor中增加Feature的資料
  - [ ] 新訓練模型

#### Method to optimize model

- change learning rate
- change limit week
- change layers parameter
- using MovieSessionData
- using MovieSessionData with One Movie One Data
- GRU replace LSTM
- Stop training when f1 not increase
- 可變長度序列嵌入到固定長度特徵向量(新研究)
- OverSampling、UnderSampling
