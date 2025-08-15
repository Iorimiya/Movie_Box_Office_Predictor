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

Test Sentiment Model with ChnSentiCorp dataset選擇：採用全部的資料集測試 or 採用 Test Set 測試

- 無論是哪一種，對Sentiment Model都是未知的
- All Dataset：接近於實際運用狀況的表現
- Test Set：SKEP有用 ChnSentiCorp ，比較Sentiment Model和SKEP的表現 <- 這個會比較好
- [ ] 兩種都存成CSV測試

## 2025/08/10

#### Finished

- [x] Continue Refactoring Prediction Code
  - [x] finish prediction data processor
  - [x] finish prediction model core
  - [x] finish prediction pipeline
  - [x] finish prediction evaluator
  - [x] fix prediction data preprocess bug
  - [x] fix prediction duplicate scaling bug
  - [x] change prediction continuous data selection
  - [x] Sentiment Model f1-score
1. 可以同時兼容Sentiment Model ( 單向的LSTM ，輸入一段文字，輸出正面或負面 )和SKEP( BERT )的架構調整
2. 順便修掉 duplicate scaling bug (Prediction model裡面的東西)
3. Sentiment Model 實作 F1-score方法
  1. 實作出 f1-score的通用方法 (判斷四個象限以及權重計算)
  2. 以1. 重新套用至prediction model (就是問題轉換成range 後套用通用方法)
  3. 以1. 建構Sentiment Model的F1-score方法
  4. 之後如果有需要，1. 可以用來建構SKEP或ELECTRA 的F1-score方法

Sentiment相關：predict -> 比對之後可以放入象限內 -> 權重計算 -> output
Prediction：predict -> 問題轉換 -> 比對之後可以放入象限內 -> 權重計算 -> output


#### TODO
- [ ] Test Sentiment Model with ChnSentiCorp dataset
  - [x] environment 衝突
- [ ] ELECTRA environment
- [ ] change base of analyse in SKEP
  - Positive -> Positive
  - Negative with score upper than 0.8 -> Negative
  - Negative with score lower or equal than 0.8 -> Positive

找到符合實際情況(ex:繁體中文)的Dataset -> 用某一組Dataset評估三種Model -> 知道每一種Model的效能 ->決定要採用什麼Model的作法(選出最強的Model、採用複合Model判斷正負、加權判斷)

總需求：我需要一個可以判斷正負面評論情緒的方法
解決總需求的方法：決定要採用什麼Model的作法

#### 目前的環境準備

- [x] Sentiment + Prediction (v0.1)
- [x] SKEP+ Prediciton (v0.2)
- [ ] Sentiment + SKEP + Prediction (v0.1.1) <-正在DEBUG
- [ ] ELECTRA +  Prediciton (v0.3)
- [ ] Sentiment + SKEP + ELECTRA +  Prediciton (v1)



