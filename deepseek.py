import pandas as pd
from model_ollama import query_model
from tqdm import tqdm

# ✅ 修复后的预测解析函数
def parse_prediction(result):
    """
    改进：从多句输出中仅取最后一句判断真/假。
    """
    if not result or not isinstance(result, str):
        return -1

    # 清洗标签
    tags_to_remove = ["<|assistant|>", "<|think|>", "</think>"]
    for tag in tags_to_remove:
        result = result.replace(tag, "")

    # 拆句：只取最后一句做判断
    lines = [line.strip() for line in result.strip().splitlines() if line.strip()]
    final_line = lines[-1] if lines else ""

    final_line = final_line.replace("：", "").replace("。", "").replace(".", "").strip().lower()

    if final_line in ["假", "虚假", "是假的", "为假", "新闻是假"]:
        return 1  # 假新闻
    elif final_line in ["真", "真实", "是真的", "为真", "新闻是真"]:
        return 0  # 真新闻
    elif "假" in final_line and "真" not in final_line:
        return 1
    elif "真" in final_line and "假" not in final_line:
        return 0
    else:
        return -1


# ✅ 任务1：直接判断真假
def classify(news_rows):
    preds = []
    for idx, row in tqdm(news_rows.iterrows(), total=len(news_rows), desc="任务1 - 判断真假"):
        prompt = f"""请判断以下新闻是真新闻还是假新闻？只回答“真”或“假”。

新闻内容：{row['content']}"""
        result = query_model(prompt)
        pred = parse_prediction(result)

        print(f"\n📰 第{idx+1}条新闻：{row['content']}")
        print(f"🤖 原始输出：{result}")
        print(f"✅ 真实标签：{'假' if row['label'] == 1 else '真'}")
        print(f"🔍 模型判断：{'假' if pred == 1 else '真' if pred == 0 else '无法判断'}")

        preds.append(pred)
    return preds

# ✅ 任务2：情感分析
def analyze_sentiment(news_rows):
    few_shot_prompt = """请判断下面新闻的情感倾向，只回答“积极”、“消极”或“中性”，不要多余文字。

示例：
新闻内容：公司业绩大幅增长，市场反响热烈。
回答：积极

新闻内容：近期天气恶劣，导致交通严重堵塞。
回答：消极

新闻内容：会议如期举行，内容正常。
回答：中性

现在判断以下新闻：
新闻内容：
"""
    sentiments = []
    for idx, row in tqdm(news_rows.iterrows(), total=len(news_rows), desc="任务2 - 分析情感"):
        prompt = few_shot_prompt + row['content'] + "\n回答："
        result = query_model(prompt)

        # 提取最后一行作为判断依据
        lines = [line.strip() for line in result.strip().splitlines() if line.strip()]
        final_line = lines[-1] if lines else ""
        final_line = final_line.replace("。", "").replace("：", "").strip()

        print(f"\n📰 第{idx+1}条新闻：{row['content']}")
        print(f"🤖 模型输出完整内容：{result}")
        print(f"🔍 最后一行判断值：{final_line}")

        if final_line == "积极":
            sentiments.append("积极")
        elif final_line == "消极":
            sentiments.append("消极")
        elif final_line == "中性":
            sentiments.append("中性")
        else:
            sentiments.append("中性")
            print("⚠️ 未能准确识别情感，默认标记为中性")
    return sentiments


# ✅ 任务3：结合情感判断真假
def classify_with_sentiment(news_rows, sentiments):
    preds = []
    for idx, ((_, row), sent) in enumerate(zip(news_rows.iterrows(), sentiments)):
        prompt = f"""这条新闻的情感是“{sent}”（供参考）。请你根据新闻内容来判断该新闻是真新闻还是假新闻？只回答“真”或“假”。

新闻内容：{row['content']}"""
        result = query_model(prompt)
        pred = parse_prediction(result)

        print(f"\n📰 第{idx+1}条新闻：{row['content']}")
        print(f"💬 情感参考：{sent}")
        print(f"🤖 模型输出：{result}")
        print(f"✅ 真实标签：{'假' if row['label'] == 1 else '真'}")
        print(f"🔍 模型判断：{'假' if pred == 1 else '真' if pred == 0 else '无法判断'}")

        preds.append(pred)
    return preds

# ✅ 准确率统计函数（跳过 -1 的结果）
def evaluate(preds, labels):
    clean_pairs = [(p, l) for p, l in zip(preds, labels) if p in [0, 1]]
    if not clean_pairs:
        return {"Accuracy": 0, "Accuracy_fake": 0, "Accuracy_true": 0}

    clean_preds, clean_labels = zip(*clean_pairs)
    correct = sum([p == l for p, l in clean_pairs])
    total = len(clean_pairs)

    correct_fake = sum([1 for p, l in clean_pairs if l == 1 and p == 1])
    total_fake = sum([1 for l in clean_labels if l == 1])

    correct_true = sum([1 for p, l in clean_pairs if l == 0 and p == 0])
    total_true = sum([1 for l in clean_labels if l == 0])

    return {
        "Accuracy": round(correct / total, 4),
        "Accuracy_fake": round(correct_fake / total_fake, 4) if total_fake else 0,
        "Accuracy_true": round(correct_true / total_true, 4) if total_true else 0
    }

# ✅ 主程序
if __name__ == "__main__":
    print("📄 加载测试集数据并抽样10条...")
    df_test = pd.read_csv("train.csv").sample(n=10, random_state=42).reset_index(drop=True)

    # 确保标签是整数
    df_test['label'] = df_test['label'].astype(int)

    print("🚀 任务1：直接判断真假新闻")
    preds1 = classify(df_test)
    eval1 = evaluate(preds1, df_test['label'].tolist())

    print("\n🔍 任务2：情感分析")
    sentiments = analyze_sentiment(df_test)

    print("\n🔁 任务3：结合情感判断真假新闻")
    preds3 = classify_with_sentiment(df_test, sentiments)
    eval3 = evaluate(preds3, df_test['label'].tolist())

    print("\n✅ 准确率汇总：")
    print("任务1（直接判断）：", eval1)
    print("任务3（结合情感）：", eval3)

    print("\n📈 提升分析：")
    print("Accuracy 提升：", round(eval3['Accuracy'] - eval1['Accuracy'], 4))
    print("Accuracy_fake 提升：", round(eval3['Accuracy_fake'] - eval1['Accuracy_fake'], 4))
    print("Accuracy_true 提升：", round(eval3['Accuracy_true'] - eval1['Accuracy_true'], 4))
