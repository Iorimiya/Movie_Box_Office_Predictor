- [X] 需要100筆的測試SKEP有效與否的資料
- [ ] model(data_processor)的完成
- [ ] 用同樣的數據對SKEP和Sentiment model做比較
- [ ] 增加評論資料中讚和虛的數量
  - [ ] 增加PublicReview的子類別PttReview
  - [ ] 在review_collector中增加蒐集的部分
  - [ ] 在WeekData中增加Feature想呈現的資料
  - [ ] 在DataProcessor中增加Feature的資料
  - [ ] 新訓練模型
  - Feature中要呈現的資料是？
          - 本週讚總數
          - 本週噓總數
          - 計算每一篇Review的讚和噓的比例，加以平均
          - 計算本週所有讚和噓的比例

#### 使用新OO架構重構程式

- [ ] finish prediction data processor
- [ ] finish prediction model core
- [ ] finish prediction pipeline
- [ ] finish prediction evaluator
- [ ] fix prediction data preprocess bug
- [ ] fix prediction duplicate scaling bug
- [ ] change prediction continuous data selection

#### SKEP

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

#### BERT

- [ ] 測試環境相容性
- [ ] 遷入主專案
- [ ] 測試程式相容性
- [ ] 建立符合BERT訓練輸入的資料集
- [ ] 訓練新prediciton model
- [ ] 評估prediction model效能

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
