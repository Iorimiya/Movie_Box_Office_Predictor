import numpy as np
import pandas as pd

# 1. 準備模擬數據 (在實際應用中替換為讀取真實數據)
weeks = 100  # 總週數 (示例數據)
np.random.seed(42)
# 模擬本週票房（已正規化的值，例如除以最大票房或以萬元為單位）
box_office = np.random.rand(weeks)  # 介於0-1之間的隨機值代表正規化票房
# 模擬觀眾評論文本資料
comment_texts = [
    "This movie was fantastic with great acting and story" if i % 2 == 0
    else "Terrible plot and poor acting, not my taste"
    for i in range(weeks)
]
# 模擬觀眾回覆數、專家評論數量、專家評論正面與否、是否有專家評論
audience_replies = np.random.randint(0, 50, size=weeks)
expert_reviews = np.random.randint(0, 5, size=weeks)
expert_positive = np.random.randint(0, 2, size=weeks)  # 0 或 1
expert_exists = (expert_reviews > 0).astype(int)       # 有沒有專家評論

# 將上述數據組成DataFrame以便處理
data = pd.DataFrame({
    'BoxOffice': box_office,
    'AudienceReplies': audience_replies,
    'ExpertReviewCount': expert_reviews,
    'ExpertPositive': expert_positive,
    'ExpertExists': expert_exists,
    'CommentText': comment_texts
})
# 顯示前幾行資料 (查看結構)
print(data.head())
# 2. NLP特徵處理：將評論文本轉換為TF-IDF特徵向量
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=50, stop_words='english')  # 限制最多50個關鍵詞特徵
tfidf_matrix = vectorizer.fit_transform(data['CommentText']).toarray()
print("TF-IDF 特徵矩陣維度:", tfidf_matrix.shape)  # (weeks, 50)

# 3. 組合數值特徵與文本特徵
numeric_features = data[['BoxOffice', 'AudienceReplies', 'ExpertReviewCount',
                         'ExpertPositive', 'ExpertExists']].values
# 將文本TF-IDF特徵與其他數值特徵在列方向合併
combined_features = np.hstack([numeric_features, tfidf_matrix])
print("每週特徵向量長度:", combined_features.shape[1])  # 總特徵數量

# 4. 資料標準化：僅以訓練集資料來擬合標準化參數
from sklearn.preprocessing import StandardScaler

# 假設我們將前80%的週作為訓練，其餘作為測試
train_weeks = int(weeks * 0.8)
train_data = combined_features[:train_weeks]
test_data = combined_features[train_weeks:]
# 用訓練資料擬合StandardScaler
scaler = StandardScaler().fit(train_data)
# 將變換應用到整個資料集（確保對訓練和測試使用相同縮放）
combined_features_scaled = scaler.transform(combined_features)

# 5. 構造LSTM序列資料：用最近4週的資料預測下一週
time_steps = 4  # 使用過去4週作為輸入
X_seq = []
y_seq = []
target_index = 0  # BoxOffice在combined_features中的列索引為0
for i in range(len(combined_features_scaled) - time_steps):
    # 取連續time_steps週的特徵作為一筆輸入
    X_seq.append(combined_features_scaled[i : i + time_steps])
    # 取下一週的票房（標準化後的值）作為預測目標
    y_seq.append(combined_features_scaled[i + time_steps, target_index])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print("序列資料 X_seq 維度:", X_seq.shape)  # (樣本數, time_steps, 特徵數)
print("對應目標 y_seq 維度:", y_seq.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 構建 LSTM 模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, X_seq.shape[2])))
# 可選：加入 Dropout 防止過擬合 (例如 model.add(Dropout(0.2)))
model.add(Dense(1))  # 輸出下一週的預測票房值
model.compile(optimizer='adam', loss='mse')
model.summary()  # 輸出模型結構摘要

# 將序列數據切分為訓練和測試集
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]
print("訓練樣本數:", X_train.shape[0], "測試樣本數:", X_test.shape[0])

# 訓練模型
history = model.fit(X_train, y_train,
                    epochs=1000, batch_size=16,
                    validation_data=(X_test, y_test),
                    verbose=2)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 使用模型對測試集做出預測
y_pred = model.predict(X_test)

# 將預測結果和實際值轉換為一維陣列（注意目前是標準化後的值）
y_pred = y_pred.reshape(-1)
y_true = y_test.reshape(-1)

# 計算評估指標
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"測試集 RMSE: {rmse:.4f}")
print(f"測試集 MAE: {mae:.4f}")
print(f"測試集 R^2: {r2:.4f}")


# 構造當前最後4週的輸入特徵（假設我們已經有最新一週的實際數據在 data 中）
last_sequence = combined_features_scaled[-time_steps:]  # 取倒數4週的標準化特徵
last_sequence = last_sequence.reshape(1, time_steps, combined_features_scaled.shape[1])
next_week_pred_scaled = model.predict(last_sequence)
# 如果需要將預測值轉回原始尺度（以萬元計），可以對標準化逆變換:
next_week_pred = next_week_pred_scaled[0, 0]
print(f"下週票房預測值（經標準化）: {next_week_pred:.4f}")
# （若知道標準化前票房的均值和方差，可反標準化以得到實際單位的預測值）


import matplotlib.pyplot as plt

# 假設我們有測試集對應的週序號
test_weeks_index = np.arange(train_weeks + time_steps, weeks)  # 從訓練集最後一週+1開始到最後
# 繪製實際值與預測值走勢
#plt.figure(figsize=(8,4))
#plt.plot(len(y_true), y_true, label='Actual', marker='o')
#plt.plot(len(y_pred), y_pred, label='Predicted', marker='x')
#plt.title('Actual vs Predicted Box Office (Test Set)')
#plt.xlabel('Week')
#plt.ylabel('Box Office (normalized)')
#plt.legend()
#plt.show()


for i in range(5):  # 這裡以前5個測試樣本為例
    actual = y_true[i]
    predicted = y_pred[i]
    print(f"週數 {train_weeks + time_steps + i} 實際值: {actual:.4f}, 預測值: {predicted:.4f}")