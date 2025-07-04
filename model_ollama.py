import requests
def query_model(prompt: str) -> str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "deepseek-r1:8b",  # ✅ 正确模型名
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # ✅ 不要用 stream:true，除非你处理 event
    }

    try:
        response = requests.post(url, json=payload)
        data = response.json()

        # ✅ 安全解析 message.content
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()

        # 如果返回了错误
        elif "error" in data:
            print("❌ 模型错误：", data["error"])
            return f"错误：{data['error']}"

        # 如果格式未知
        else:
            print("❌ 未知响应结构：", data)
            return "未知响应结构"

    except Exception as e:
        print("❌ JSON 解析失败：", e)
        print("状态码：", response.status_code)
        print("原始响应：", response.text)
        return "解析失败"
