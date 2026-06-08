"""海上风电历史数据分析与多算法智能体 — 后端入口 (Flask API Server)
=============================================================
提供 RESTful API, 供 Vue 前端调用:
  - GET  /api/dashboard       总览大屏数据 (含验证集精度分析)
  - GET  /api/site/<site_id>  站点详情
  - GET  /api/comparison      模型/算法对比实验数据
  - POST /api/ask             智能问答
  - GET  /api/export/*        数据导出
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_file

from wind_agent import OffshoreWindAgentSystem

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 创建 Flask 应用
app = Flask(__name__)

# 初始化海上风电智能体系统 (serve 模式: 仅加载最优历史结果, 绝不触发训练)
# 如果没有训练过, 启动时会报错并提示先运行 python train_system.py
system = OffshoreWindAgentSystem(PROJECT_ROOT, mode="serve")


# ============================================================================
# CORS 中间件 — 允许前端跨域访问
# ============================================================================

@app.after_request
def add_cors_headers(response):
    """为所有响应添加 CORS 头, 允许前端跨域请求"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ============================================================================
# API 路由
# ============================================================================

@app.route("/", methods=["GET"])
def health():
    """健康检查 / 根路径"""
    return jsonify({
        "name": "Offshore Wind Agent Backend",
        "status": "ok",
        "version": system.version,
        "message": "海上风电历史数据分析系统已就绪。请使用 Vue 前端访问 Dashboard。",
    })


@app.route("/api/<path:_path>", methods=["OPTIONS"])
def api_options(_path: str):
    """处理预检请求 (CORS preflight)"""
    return ("", 204)


@app.route("/api/dashboard")
def dashboard():
    """获取 Dashboard 总览页的全量数据

    包含: 汇总指标卡片、站点概览、模型指标、RL策略对比、
          实验模块、高风险窗口、元信息、问答后端状态等。
    """
    return jsonify(system.get_dashboard_payload())


@app.route("/api/site/<site_id>")
def site_detail(site_id: str):
    """获取指定站点的详情数据

    包含: 预测时序、风险曲线、日汇总、高风险窗口、调度建议等。
    """
    return jsonify(system.get_site_payload(site_id))


@app.route("/api/comparison")
def comparison():
    """获取模型对比和算法对比数据 (新增)

    包含:
      - model_comparison: HistGBR vs LightGBM 预测精度对比
      - rl_algorithm_comparison: Q-Learning vs DQN 调度策略对比
      - model_metrics: 完整的模型评估指标
      - rl_comparison: 所有调度策略的横向对比
    """
    payload = system.get_dashboard_payload()
    return jsonify({
        "model_comparison": payload.get("model_comparison", {}),
        "rl_algorithm_comparison": payload.get("rl_algorithm_comparison", {}),
        "model_metrics": payload.get("model_metrics", {}),
        "rl_comparison": payload.get("rl_comparison", []),
        "experiment_modules": payload.get("experiment_modules", []),
        "algorithm_simulation": payload.get("algorithm_simulation", {}),
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    """智能问答接口

    接收 JSON: {"question": "..."}
    返回 JSON: {"question": "...", "answer": "...", "backend": "ollama"|"rule-based", "model": "..."}

    优先调用 Ollama LLM, 不可用时回退到内置规则问答。
    """
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400
    return jsonify(system.answer_question(question))


# ============================================================================
# 动态数据加载接口
# ============================================================================

@app.route("/api/upload/predict", methods=["POST"])
def upload_predict():
    """上传测试集 CSV 文件, 返回预测结果 JSON (供前端展示)

    请求: multipart/form-data, 字段名 "file"
    返回: JSON { columns: [...], rows: [[...], ...], total: N }
    """
    if "file" not in request.files:
        return jsonify({"error": "请上传 CSV 文件, 字段名: file"}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "仅支持 .csv 格式"}), 400

    try:
        from io import BytesIO
        import pandas as pd
        csv_bytes = BytesIO(uploaded.read())
        result_buffer = system.predict_on_upload(csv_bytes)
        # 解析为 JSON
        result_buffer.seek(0)
        df = pd.read_csv(result_buffer)
        resp = {
            "filename": uploaded.filename,
            "columns": list(df.columns),
            "rows": df.values.tolist(),
            "total": len(df),
        }
        # 如果结果包含实际功率列, 附加精度指标
        if "实际功率_MW" in df.columns:
            from sklearn.metrics import mean_absolute_error, r2_score
            import numpy as np
            actual = df["实际功率_MW"].dropna()
            predicted = df["预测功率_MW"].loc[actual.index]
            resp["has_actual"] = True
            resp["mae_mw"] = round(float(mean_absolute_error(actual, predicted)), 3)
            resp["r2"] = round(float(r2_score(actual, predicted)), 4)
            resp["rmse_mw"] = round(float(np.sqrt(((actual - predicted) ** 2).mean())), 3)
        return jsonify(resp)
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"[Upload Error] {err_detail}", flush=True)
        return jsonify({
            "error": f"预测失败: {str(e)}",
            "debug": {
                "loaded_run": system.run_info.get("run_name", "unknown"),
                "model_features": len(system.forecast_bundle.model.feature_names_in_) if system.forecast_bundle else "N/A",
            }
        }), 500


@app.route("/api/upload/predict/download", methods=["POST"])
def upload_predict_download():
    """上传 CSV 并直接下载预测结果文件"""
    if "file" not in request.files:
        return jsonify({"error": "请上传 CSV 文件"}), 400
    uploaded = request.files["file"]
    try:
        from io import BytesIO
        csv_bytes = BytesIO(uploaded.read())
        result = system.predict_on_upload(csv_bytes)
        return send_file(
            result, as_attachment=True,
            download_name=f"predicted_{uploaded.filename}",
            mimetype="text/csv",
        )
    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500


@app.route("/api/download/template")
def download_template():
    """下载示例测试集 CSV 模板 (不含实际功率, 用于测试上传接口)

    返回: 默认测试集文件 (test_weather.csv), 去掉出力(MW)列
    """
    template_path = PROJECT_ROOT / "data" / "raw" / "test_weather.csv"
    if not template_path.exists():
        return jsonify({"error": "模板文件不存在"}), 404
    return send_file(
        str(template_path),
        as_attachment=True,
        download_name="test_template.csv",
        mimetype="text/csv",
    )


@app.route("/api/debug/info")
def debug_info():
    """诊断端点: 返回当前加载的模型信息"""
    fb = system.forecast_bundle
    model_feats = list(fb.model.feature_names_in_) if fb else []
    return jsonify({
        "loaded_run": system.run_info.get("run_name", "unknown"),
        "model_type": type(fb.model).__name__ if fb else "N/A",
        "model_features_count": len(model_feats),
        "model_features": model_feats,
        "bundle_features_count": len(fb.feature_columns) if fb else 0,
        "dqn_agent_loaded": system.dqn_agent is not None,
        "rl_comparison": system.summary_payload.get("rl_algorithm_comparison", {}).get("comparison", {}) if system.summary_payload else {},
    })


# ============================================================================
# 上传测试集管理 API (持久化, 跨页面共享)
# ============================================================================

@app.route("/api/upload/save", methods=["POST"])
def upload_save():
    """保存预测结果到磁盘 (在 upload/predict 之后调用)

    请求: JSON { csv_content: "...", original_filename: "..." }
    返回: JSON 元信息 { id, original_filename, sites, total_rows, ... }
    """
    payload = request.get_json(silent=True) or {}
    csv_content = payload.get("csv_content", "")
    original_filename = payload.get("original_filename", "unknown.csv")

    if not csv_content:
        return jsonify({"error": "缺少 csv_content"}), 400

    try:
        from io import BytesIO
        buf = BytesIO(csv_content.encode("utf-8-sig"))
        meta = system.save_upload(buf, original_filename)
        return jsonify(meta)
    except Exception as e:
        return jsonify({"error": f"保存失败: {str(e)}"}), 500


@app.route("/api/uploads")
def list_uploads():
    """列出所有已保存的上传记录 (按时间倒序)"""
    return jsonify(system.list_uploads())


@app.route("/api/uploads/<upload_id>")
def upload_detail(upload_id: str):
    """获取单次上传的详情 (含前200行预览)"""
    detail = system.get_upload_detail(upload_id)
    if detail is None:
        return jsonify({"error": "上传记录不存在"}), 404
    return jsonify(detail)


@app.route("/api/uploads/<upload_id>/site/<site_id>")
def upload_site_data(upload_id: str, site_id: str):
    """获取上传数据中指定站点的完整时序数据"""
    data = system.get_upload_site_data(upload_id, site_id)
    if data is None:
        return jsonify({"error": "上传记录或站点不存在"}), 404
    return jsonify(data)


@app.route("/api/uploads/<upload_id>", methods=["DELETE"])
def delete_upload(upload_id: str):
    """删除指定上传记录"""
    ok = system.delete_upload(upload_id)
    if not ok:
        return jsonify({"error": "上传记录不存在"}), 404
    return jsonify({"deleted": upload_id})


@app.route("/api/export/forecast.csv")
def export_forecast():
    """导出测试集预测结果为 CSV 文件

    包含字段: site_id, timestamp, predicted_power_mw, risk_score,
             recommended_action, recommended_dispatch_mw, reserve_factor
    """
    return send_file(
        system.export_forecast_csv(),
        as_attachment=True,
        download_name="offshore_wind_forecast.csv",
        mimetype="text/csv",
    )


@app.route("/api/export/rl_report.json")
def export_rl_report():
    """导出强化学习完整报告为 JSON 文件

    包含: 系统摘要、模型对比、RL策略对比、实验模块、样例数据等。
    """
    return send_file(
        system.export_rl_report_json(),
        as_attachment=True,
        download_name="offshore_wind_rl_report.json",
        mimetype="application/json",
    )


# ============================================================================
# 启动入口
# ============================================================================

if __name__ == "__main__":
    # debug=False 适用于生产/演示环境
    app.run(host="127.0.0.1", port=5000, debug=False)
