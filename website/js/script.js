/**
 * 處理前端與後端通訊的腳本
 */

document.addEventListener('DOMContentLoaded', () => {
    const submitBtn = document.querySelector('.inputGroup button');
    const inputField = document.getElementById('movieQuote');
    const traceBox = document.getElementById('traceBox');
    const statusDiv = document.querySelector('.result .status');

    if (submitBtn) {
        submitBtn.addEventListener('click', async () => {
            // 1. 準備請求資料
            // 雖然是測試 TensorFlow 環境，但我們還是把輸入框的內容帶過去，證明參數傳遞正常
            const userInput = inputField.value;
            
            const requestPayload = {
                command: "verify_tensorflow",
                arguments: {
                    user_input_movie: userInput,
                    test_mode: true
                }
            };

            // 2. 更新 UI 狀態
            traceBox.innerHTML = "正在連線至計算伺服器...";
            statusDiv.innerText = "處理中...";
            submitBtn.disabled = true;

            try {
                // 3. 發送請求
                // Nginx 會將 /api/compute 代理到後端的 /compute
                const response = await fetch('/api/compute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestPayload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                // 4. 解析回應
                const data = await response.json();
                console.log("收到後端回應:", data);

                // 5. 顯示結果
                if (data.status === 'completed') {
                    statusDiv.innerText = "執行成功";
                    // 將結果格式化顯示在 traceBox
                    traceBox.innerHTML = `<pre>${JSON.stringify(data.result, null, 2)}</pre>`;
                } else {
                    statusDiv.innerText = "執行失敗";
                    traceBox.innerText = `錯誤: ${data.error || '未知錯誤'}`;
                }

            } catch (error) {
                console.error("請求失敗:", error);
                statusDiv.innerText = "連線錯誤";
                traceBox.innerText = `連線失敗: ${error.message}`;
            } finally {
                submitBtn.disabled = false;
            }
        });
    }
});