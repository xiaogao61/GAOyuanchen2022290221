import pandas as pd
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from model_ollama import query_model  # 使用你已有的本地大模型接口


# ✅ 设置中文显示（解决热力图坐标乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv("train.csv").sample(n=10, random_state=42)
texts = df["content"].astype(str).tolist()

# 2. 中文分词与清洗
def preprocess(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)  # 保留中文
    tokens = jieba.cut(text)
    return " ".join([t for t in tokens if len(t) > 1])

texts_cleaned = [preprocess(text) for text in texts]

# 3. 文本向量化（词袋模型）
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts_cleaned)

# 4. LDA建模
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# 5. 打印每个主题的关键词
def print_topics(model, feature_names, n_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"主题 {idx + 1}: ", " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_topics(lda, vectorizer.get_feature_names_out(), 10)

# 6. 词云图可视化
for topic_idx, topic in enumerate(lda.components_):
    freqs = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[:-30 - 1:-1]}
    wc = WordCloud(font_path='simhei.ttf', background_color='white', width=800, height=400)
    wc.generate_from_frequencies(freqs)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"主题 {topic_idx + 1} 词云")
    plt.show()


# 8. 热力图：文档-主题分布
doc_topic_matrix = lda.transform(X)
df_doc_topic = pd.DataFrame(doc_topic_matrix, columns=[f"主题{i+1}" for i in range(lda.n_components)])
df_doc_topic.index = [f"微博{i+1}" for i in range(len(texts))]

plt.figure(figsize=(10, 6))
sns.heatmap(df_doc_topic, cmap="YlOrRd", annot=True)
plt.title("📊 每篇微博的主题概率热力图")
plt.xlabel("主题")
plt.ylabel("微博")
plt.tight_layout()
plt.show()

# 9. 调用本地大模型分析每个主题的语义含义
feature_names = vectorizer.get_feature_names_out()
topn = 10
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-topn - 1:-1]]
    prompt = f"""以下是微博主题模型中第{topic_idx + 1}个主题的关键词：
{', '.join(top_words)}

请你根据这些关键词推测这个主题主要在讨论什么内容？请用简洁中文总结，不超过50字。"""
    result = query_model(prompt)
    print(f"\n🧠 主题 {topic_idx + 1} 关键词：{top_words}")
    print(f"🤖 大模型分析结果：{result.strip()}")
