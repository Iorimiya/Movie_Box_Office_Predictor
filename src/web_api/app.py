import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import tensorflow as tf
from flask import Flask, jsonify, request


@dataclass
class ComputeRequest:
    command: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'ComputeRequest':
        """將 JSON 字串反序列化為 Python 物件"""
        try:
            data = json.loads(json_str)
            return cls(
                command=data.get("command", ""),
                arguments=data.get("arguments", {})
            )
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")


@dataclass
class ComputeResponse:
    status: str  # 'in_progress', 'completed', 'failed'
    result: Optional[dict[str, Any]] = None
    progress: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None

    def to_json(self) -> str:
        """將 Python 物件序列化為 JSON 字串"""
        return json.dumps(asdict(self), ensure_ascii=False)


# 設定簡易 Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def verify_tensorflow_environment() -> dict[str, Any]:
    """
    執行簡單的 TensorFlow 環境驗證計算。
    """
    try:
        logger.info("Starting TensorFlow verification...")

        # 取得版本
        version = tf.__version__

        # 檢查 GPU
        gpu_devices = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpu_devices) > 0
        gpu_details = [str(d) for d in gpu_devices]

        # 執行一個簡單的張量運算測試
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)

        return {
            "version": version,
            "gpu_available": gpu_available,
            "gpu_details": gpu_details,
            "tensor_calculation_test": f"Matmul result shape: {c.shape}",
            "operation_success": True
        }
    except Exception as e:
        logger.error(f"TensorFlow verification failed: {e}")
        return {
            "operation_success": False,
            "error": str(e)
        }


@app.route('/compute', methods=['POST'])
def compute():
    """
    接收前端請求的端點。
    對應 Nginx 設定: /api/compute -> http://<host>:11100/compute
    """
    try:
        # 1. 解析請求 JSON
        raw_data = request.get_data(as_text=True)
        if not raw_data:
            resp = ComputeResponse(status="failed", error={"message": "No JSON data received"})
            return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

        try:
            req_obj = ComputeRequest.from_json(raw_data)
        except ValueError as e:
            resp = ComputeResponse(status="failed", error={"message": str(e)})
            return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

        logger.info(f"Received command: {req_obj.command} with args: {req_obj.arguments}")

        # 2. 根據 command 執行對應動作 (目前只實作 verify_tensorflow)
        if req_obj.command == "verify_tensorflow":
            # 模擬進度 (實際專案可透過 WebSocket 或輪詢實作)
            # 這裡直接回傳最終結果
            result = verify_tensorflow_environment()

            response_obj = ComputeResponse(
                status="completed",
                progress={
                    "current_step": 1,
                    "total_steps": 1,
                    "message": "Verification done"
                },
                result=result
            )
            return app.response_class(response=response_obj.to_json(), status=200, mimetype='application/json')

        resp = ComputeResponse(status="failed", error={"message": f"Unknown command: {req_obj.command}"})
        return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

    except Exception as e:
        logger.error(f"Server error: {e}")
        resp = ComputeResponse(status="failed", error={"message": str(e)})
        return app.response_class(response=resp.to_json(), status=500, mimetype='application/json')


if __name__ == '__main__':
    # 監聽 0.0.0.0 以允許外部連線 (Nginx 代理)
    # Port 設定為 11100 以配合 Nginx 設定
    app.run(host='0.0.0.0', port=11100, debug=True)
