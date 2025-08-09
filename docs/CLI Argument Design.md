
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
