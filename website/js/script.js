/**
 * CinePulse 電影分析儀表板前端邏輯
 */

const MOVIE_DATA_MOCK_LIST = [
    { id: 101, name: "昨日的雨", buzz: "4,511" },
    { id: 102, name: "光之邊界", buzz: "4,389" },
    { id: 103, name: "最後一班車", buzz: "4,377" },
    { id: 104, name: "無聲片段", buzz: "4,236" },
    { id: 105, name: "愛的迴路", buzz: "3,760" },
    { id: 108, name: "逆轉時刻", buzz: "4,712" }
];

let distChart = null;

/**
 * 渲染首頁熱度清單
 */
function renderHome() {
    const listHot = document.getElementById('listHot');
    if (!listHot) return;

    listHot.innerHTML = [...MOVIE_DATA_MOCK_LIST]
        .sort((a, b) => parseInt(b.buzz.replace(/,/g, '')) - parseInt(a.buzz.replace(/,/g, '')))
        .map(m => `
            <div onclick="openMovie('${m.name}')" class="list-hover px-4 py-4 flex justify-between">
                <div class="font-semibold">${m.name}</div>
                <div class="font-bold">${m.buzz}</div>
            </div>
        `).join('');
}

/**
 * 從後端獲取詳細電影資料並切換頁面
 * @param {string} movieName 電影中文名稱
 */
async function openMovie(movieName) {
    console.log(`正在請求電影資料: ${movieName}`);

    const requestPayload = {
        command: "get_movie_data",
        arguments: {
            movie_name: movieName
        }
    };

    try {
        // Nginx 代理後的 API 路徑
        const response = await fetch('/api/compute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestPayload)
        });

        if (!response.ok) throw new Error("API 請求失敗");

        const data = await response.json();

        if (data.status === 'completed' && data.result) {
            const movie = data.result;

            // 1. 更新基本資訊
            document.getElementById('movieTitle').innerText = movie.chinese_name || movieName;
            document.getElementById('movieDirector').innerText = movie.director || "未知";
            document.getElementById('movieScenarist').innerText = movie.scenarist || "未知";
            document.getElementById('movieIntroduction').innerText = movie.introduction || "暫無介紹";
            document.getElementById('detailBuzz').innerText = movie.num_of_reviews.toLocaleString();
            document.getElementById('detailBO').innerText = "NT$ " + movie.box_office_amount.toLocaleString();

            // 2. 更新演員 (陣列處理)
            const actorsDiv = document.getElementById('movieActors');
            actorsDiv.innerHTML = movie.actors && movie.actors.length > 0
                ? movie.actors.join('<br>')
                : "資料讀取中";

            // 3. 更新 AI 預測星等 (數值轉星等)
            const starsDiv = document.getElementById('moviePredictionStars');
            const starCount = movie.predict_result || 0;
            starsDiv.innerText = "⭐ ".repeat(starCount) || "（評估中）";

            // 4. 更新評論容器 (由後端驅動渲染)
            const reviewsContainer = document.getElementById('movieReviewsContainer');
            reviewsContainer.innerHTML = movie.reviews && movie.reviews.length > 0
                ? movie.reviews.map(content => `
                    <div class="card p-6">
                        <div class="text-sm text-slate-400 mb-2">網友評論</div>
                        <div class="bg-slate-100 rounded-xl p-4 text-slate-700">
                            ${content}
                        </div>
                    </div>
                `).join('')
                : '<div class="text-center text-slate-400">暫無相關評論</div>';

            // 5. 切換頁面
            document.getElementById('page-home').classList.add('hidden');
            document.getElementById('page-movie').classList.remove('hidden');

            // 6. 更新圖表
            initChart();

        } else {
            alert(`載入失敗: ${data.error ? data.error.message : '未知錯誤'}`);
        }
    } catch (error) {
        console.error("Error fetching movie data:", error);
        alert("連線到伺服器時發生錯誤，請稍後再試。");
    }
}

/**
 * 初始化 Chart.js 圖表
 */
function initChart() {
    const ctx = document.getElementById('chartDist');
    if (!ctx) return;

    if (distChart) distChart.destroy();

    distChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0', '0.25', '0.5', '0.75', '1'],
            datasets: [{
                data: [45, 75, 80, 45, 15],
                backgroundColor: '#93c5fd'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}

// 綁定回首頁按鈕
document.addEventListener('DOMContentLoaded', () => {
    const btnBack = document.getElementById('btnBack');
    if (btnBack) {
        btnBack.onclick = () => {
            document.getElementById('page-home').classList.remove('hidden');
            document.getElementById('page-movie').classList.add('hidden');
        };
    }

    renderHome();
});
