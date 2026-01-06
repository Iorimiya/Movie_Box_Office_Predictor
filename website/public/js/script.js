/**
 * Movie Predictor Controller
 */

const APP_CONFIG = {
    API_BASE_URL: "../src/"
};

class MoviePredictor {
    /**
     * @param {string} inputId
     * @param {string} buttonSelector
     * @param {string} statusSelector
     * @param {string} traceBoxId
     */
    constructor({ inputId, buttonSelector, statusSelector, traceBoxId }) {
        this.input = document.getElementById(inputId);
        this.button = document.querySelector(buttonSelector);
        this.status = document.querySelector(statusSelector);
        this.traceBox = document.getElementById(traceBoxId);

        this.init();
    }

    /**
     * Initialize event listeners
     */
    init() {
        this.button.addEventListener("click", () => this.onSubmit());
        this.input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") this.onSubmit();
        });
    }

    /**
     * Update the trace box with log messages
     * @param {string[]} lines
     */
    setTrace(lines) {
        this.traceBox.textContent = lines.join("\n");
    }

    /**
     * Call the backend PHP API
     * @param {string} movie
     * @returns {Promise<Object>}
     */
    async callPHP({ movie }) {
        const apiUrl = `${APP_CONFIG.API_BASE_URL}api.php?action=dummy`;

        this.setTrace([
            "JS: callPHP() 開始",
            `JS: fetch → ${apiUrl}`
        ]);

        const response = await fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `movie=${encodeURIComponent(movie)}`
        });

        this.setTrace([
            "JS: fetch 已送出",
            `JS: HTTP Status = ${response.status}`
        ]);

        const data = await response.json();

        this.setTrace([
            "JS: JSON 解析完成",
            `PHP: ${data.trace}`
        ]);

        return data;
    }

    /**
     * Handle form submission
     */
    async onSubmit() {
        const movieName = this.input.value.trim();

        this.setTrace(["JS: onSubmit() 觸發"]);

        if (!movieName) {
            this.status.textContent = "請輸入電影名稱";
            this.setTrace([
                "JS: onSubmit() 觸發",
                "JS: movie 為空，停止"
            ]);
            return;
        }

        this.status.textContent = "預測中…";
        this.button.disabled = true;

        try {
            const result = await this.callPHP({ movie: movieName });
            this.status.textContent = `${result.pr_class}（${result.stars} 星）`;

            this.setTrace([
                "JS: onSubmit() 完成",
                `JS: PR = ${result.pr_class}`,
                `PHP: ${result.trace}`
            ]);
        } catch (error) {
            this.status.textContent = "發生錯誤";
            this.setTrace([
                "JS: 發生錯誤",
                String(error)
            ]);
        } finally {
            this.button.disabled = false;
        }
    }
}

// 實例化
document.addEventListener("DOMContentLoaded", () => {
    new MoviePredictor({
        inputId: "movieQuote",
        buttonSelector: ".inputGroup button",
        statusSelector: ".status",
        traceBoxId: "traceBox"
    });
});
