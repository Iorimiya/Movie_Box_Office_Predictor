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

---

top_parser

- dataset
  - index
    --structured-dataset-name:str (required) 
    func: create_dataset_index() 
  - collect
    - box_office -> 
      --structured-dataset-name:str or --movie_name:str (mutually_exclusive)(required)
      func: collect_box_office()
    - ptt_review -> 
      --structured-dataset-name:str or --movie_name:str (mutually_exclusive)(required)
      func: collect_ptt_review()
    - dcard_review -> 
      --structured-dataset-name:str or --movie_name:str (mutually_exclusive)(required)
      func: collect_dcard_review()
  - compute_sentiment -> 
    --structured-dataset-name:str (required), 
    --model_id:str (required)
    --epoch:int (required)

- sentiment_score_model
  - train
    --feature-dataset-name:str or --structured-dataset-name:str or --random-data:flag (mutually_exclusive) (required), 
    --model-id:str (required), 
    --old-epoch:int (optional), 
    --target-epoch:int (required), 
    --checkpoint-interval:int (optional)
    func: train_sentiment_score_model() 
  - test
    --input-sentence:str (required)
    --model-id:str (required)
    --epoch:int (required)
    func: test_sentiment_score_model()
  - evaluate
    - plot
      --training-loss:flag
      --validation-loss:flag
      --f1-score:flag
      (require at least 1 flag above)
      --model-id:str  (required)
      func: plot_graph_of_sentiment_score_model()
    - get-metrics
      --training-loss:flag
      --validation-loss:flag
      --f1-score:flag
      (require at least 1 flag above)
      --model-id:str (required)
      --epoch:int (required)
      func: get_metrics_of_sentiment_score_model()

- prediction_model
  - train
    --feature-dataset-name:str or --structured-dataset-name:str or --random-data:flag (mutually_exclusive) (required), 
    --model-id:str (required), 
    --old-epoch:int (optional), 
    --target-epoch:int (required), 
    --checkpoint-interval:int (optional)
    func: train_prediction_model() 
  - test
    --movie_name:str or --random:flag (mutually_exclusive) (required)
    --model-id:str (required)
    --epoch:int (required)
    func: test_prediction_model()
  - evaluate
    - plot
      --training-loss:flag
      --validation-loss:flag
      --f1-score:flag
      (require at least 1 flag above)
      --model-id:str  (required)
      func: plot_graph_of_prediction_model()
    - get-metrics
      --training-loss:flag
      --validation-loss:flag
      --f1-score:flag
      (require at least 1 flag above)
      --model-id:str (required)
      --epoch:int (required)
      
      func: get_metrics_of_prediction_model()