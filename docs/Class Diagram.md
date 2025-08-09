```mermaid
---
title: Data Structure
---
classDiagram
    class BoxOffice{
        +date Start_date
        +date end_date
        +int box_office
    }
    class Review{
        +str url
        +str title
        +str content
        +date date
        +float? sentiment_score
        /sentiment_score_v1() bool?
    }
    class PublicReview{
        +int reply_count
    }
    class ExpertReview{
        +float expert_score
    }
    class WeekData{
        +BoxOffice box_office_data
        +List[PublicReview] public_reviews
        +List[ExpertReview] expert_reviews
        /public_review_count() int
        /expert_review_count() int
        /review_count() int
        /reply_count() int
%%        /average_sentiment_score() float
        /positive_review_count() int
        /negative_review_count() int
        %% 中性數量()
        /average_expert_score() float
    }
    class MovieData{
        +list[BoxOffice] box_office
        +list[PublicReview] public_reviews
        +list[ExpertReview] expert_reviews
    }
    class MovieSessionData{
        +list[WeekData] weeks_data
    }
    class Dataset{
        
    }
    PublicReview --|> Review
    ExpertReview --|> Review
    BoxOffice --o WeekData
    PublicReview --o WeekData
    ExpertReview --o WeekData
    WeekData --o MovieSessionData
    PublicReview --o MovieData
    ExpertReview --o MovieData
  
    Dataset ..> MovieData
    Dataset ..> MovieSessionData
```

```mermaid
classDiagram
    class Feature{
        +int box_office
        +int review_count
        +int positive_review_count
        +int negative_review_count
        +int 中性數量
        +int reply_count
    }
    
    class Timestamp{
        +list[Feature] feature
    }
    class Sample{
        +list[Timestamp] timestamp
    }
    Feature --o Timestamp
    Timestamp --o Sample
```


```python
positive = len(review for review in public_reviews if review.sentiment_score > 0)
negative =len(review for review in public_reviews if review.sentiment_score < 0)

reply_count = sum(review.reply_count for review in public_reviews)

```
sentiment_score
- v1
    sentiment model
    0 < score < 1
    (score > 0.5) => true
    (score <= 0.5) => false
- v2 (SKEP)
    正面or負面 + 預測準確率 (0~1)
    正面 = 1、負面 = -1
    (1 or -1) x 準確率
    1~0 、 -1 ~ 0 => -1 ~ 1
    (score > 0) => true
    (score <= 0) => false
    if 準確率 = 0
