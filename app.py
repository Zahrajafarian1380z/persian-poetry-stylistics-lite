import os
import re
import matplotlib
matplotlib.use('Agg')  # جلوگیری از ارورهای نمایش روی سرورهای لینوکس
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import arabic_reshaper
from bidi.algorithm import get_display

# تنظیم فونت استاندارد برای پشتیبانی از فارسی در گراف
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------------------
# ۱. بانک کلمات و ثوابت
# ------------------------------------------------------------------------------
STOPWORDS = set([
    "و", "به", "از", "ز", "که", "کز", "را", "بر", "با", "تا", "برای", "در",
    "این", "آن", "همین", "همان", "یک", "دو", "سه", "چند", "هر", "همه", "نه",
    "نیست", "باش", "بود", "شد", "شود", "می", "رو", "دگر", "دیگر", "چو", "چون",
    "گر", "اگر", "لیک", "اما", "پس", "تو", "او", "ما", "شما", "ایشان", "آنها",
    "خود", "هم", "نیز", "حتا", "جز", "غیر", "مثل", "مانند", "چنین", "چنان",
    "کجا", "چرا", "چگونه", "کی", "کدام", "آیا", "ای", "یا", "چه", "است", "بودن", "شدن"
])

STYLE_GOLDEN_WORDS = {
    "عراقی": {
        "عشق", "معشوق", "عاشق", "دل", "جان", "پیر", "رند", "زلف", "ساقی", "جام",
        "شمع", "پروانه", "خرابات", "صوفی", "باده", "فراق", "وصال", "غم", "نگار",
        "میکده", "رخ", "نظر", "آستان", "طریقت", "حقیقت", "شاهد", "مطرب", "درویش",
        "صنم", "بت", "میخانه", "مغان", "خراب", "حریف", "قدح", "سبو", "خم",
        "مست", "مستی", "سوز", "گداز", "اشک", "ناله", "حسرت", "وفا", "جفا",
        "جور", "ناز", "نیاز", "درد", "درمان", "طبیب", "بلا", "فتنه", "کوی",
        "دیوانه", "مجنون", "لیلی", "سودا", "هوس", "شوق", "امید", "خال", "خط",
        "لب", "چشم", "ابرو", "قمت", "سرو", "یاقوت", "پیمانه", "ساغر", "صحبت",
        "خلوت", "انس", "فنا", "بقا", "سالک", "مرشد", "طاهر", "پاک", "گناه",
        "توبه", "زهد", "صومعه", "محراب", "مسجد", "غزل", "نغمه", "چنگ", "رباب",
        "سجاده", "طاعت", "خرقه", "سالوس", "کرم", "لطف", "عنایت", "هجران", "دیدار"
    },
    "خراسانی": {
        "میهن", "شاه", "جنگ", "تیغ", "سپاه", "خسرو", "رزم", "بزم", "دژ", "اسپ",
        "پهلوان", "گرد", "تاج", "تخت", "یزدان", "دیو", "کمند", "زوبین", "نامور",
        "جهاندار", "کاووس", "رستم", "سهراب", "درفش", "سپهبد", "شیر", "اژدها"
    },
    "هندی": {
        "مضمون", "خیال", "آینه", "حیرت", "تمثیل", "موج", "گوهر", "تنگ", "پیچیده",
        "شیشه", "شرار", "نگاه", "رمزدان", "نازک", "معنی", "حباب", "دستگاه", "سرمه"
    },
    "معاصر": {
        "شب", "روز", "شهر", "کوچه", "پنجره", "باران", "امید", "تنهایی", "غربت",
        "آفتاب", "سایه", "دیوار", "سفر", "راه", "خیابان", "مرگ", "زندگی", "مردم"
    }
}

def preprocess_text(text):
    text = re.sub(r'[\u200c\u200b\u200e\u200f\u202a-\u202e]', ' ', text)
    replacements = {'ي': 'ی', 'ى': 'ی', 'ئ': 'ی', 'ك': 'ک', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و'}
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    return re.sub(r"[ًٌٍَُِّْٰٖٓٔـ]", "", text)

def fix_persian_text(text):
    """اصلاح جهت و اتصال حروف فارسی برای رسم در Matplotlib"""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

# ------------------------------------------------------------------------------
# ۲. تابع پردازش و تحلیل سبک
# ------------------------------------------------------------------------------
def analyze_style_pipeline(raw_text, file_obj, selected_style, top_k_words):
    if file_obj is not None:
        try:
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception as e:
            return None, pd.DataFrame({"خطا": [f"مشکل در خواندن فایل: {str(e)}"]})

    if not raw_text or not raw_text.strip():
        return None, pd.DataFrame({"پیام": ["لطفاً یک متن وارد کنید یا فایل متنی آپلود نمایید."]})

    clean_txt = preprocess_text(raw_text)
    sentences = [s.strip() for s in clean_txt.splitlines() if s.strip()]

    cleaned_docs = []
    for s in sentences:
        words = [w for w in s.split() if w not in STOPWORDS and len(w) > 1]
        if words:
            cleaned_docs.append(" ".join(words))

    if not cleaned_docs:
        return None, pd.DataFrame({"پیام": ["واژه معتبری پس از حذف استاپ‌وردها یافت نشد."]})

    vectorizer = TfidfVectorizer(max_features=300, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    vocab = list(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    G_cooc = nx.Graph()
    for doc in cleaned_docs:
        words = doc.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 3, len(words))):
                w1, w2 = words[i], words[j]
                if w1 != w2:
                    G_cooc.add_edge(w1, w2, weight=G_cooc.get_edge_data(w1, w2, {}).get('weight', 0) + 1)

    word_doc_matrix = tfidf_matrix.T.toarray()
    sim_matrix = cosine_similarity(word_doc_matrix)

    G_semantic = nx.Graph()
    n = len(vocab)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim > 0.2:
                G_semantic.add_edge(vocab[i], vocab[j], weight=float(sim))

    pr_cooc = nx.pagerank(G_cooc, weight='weight', max_iter=50) if len(G_cooc) > 0 else {w: 0 for w in vocab}
    pr_semantic = nx.pagerank(G_semantic, weight='weight', max_iter=50) if len(G_semantic) > 0 else {w: 0 for w in vocab}

    df = pd.DataFrame({'keyword': vocab, 'tfidf': tfidf_scores})
    df['pagerank_cooc'] = df['keyword'].map(pr_cooc).fillna(0)
    df['pagerank_bert'] = df['keyword'].map(pr_semantic).fillna(0)

    def min_max_norm(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    df['norm_tfidf'] = min_max_norm(df['tfidf'])
    df['norm_pr_cooc'] = min_max_norm(df['pagerank_cooc'])
    df['norm_pr_bert'] = min_max_norm(df['pagerank_bert'])

    golden_set = STYLE_GOLDEN_WORDS.get(selected_style, set())
    df['style_modifier'] = df['keyword'].apply(lambda w: 1.4 if w in golden_set else 0.75)
    df['status'] = df['keyword'].apply(lambda w: "طلایی (سبکی)" if w in golden_set else "جریمه‌شده")

    df['composite_score'] = (
        0.30 * df['norm_tfidf'] +
        0.35 * df['norm_pr_cooc'] +
        0.35 * df['norm_pr_bert']
    ) * df['style_modifier']

    df_top = df.sort_values(by='composite_score', ascending=False).head(int(top_k_words)).copy()

    # رسم گراف با اصلاح کامل عبارات فارسی
    G_viz = nx.Graph()
    main_node = "CENTER"
    G_viz.add_node(main_node)

    node_label_map = {}
    for _, row in df_top.iterrows():
        fixed_kw = fix_persian_text(row['keyword'])
        G_viz.add_edge(main_node, fixed_kw, weight=row['composite_score'])
        node_label_map[fixed_kw] = fixed_kw

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    pos = nx.spring_layout(G_viz, k=0.55, iterations=40, seed=42)

    nx.draw_networkx_nodes(G_viz, pos, nodelist=[main_node], node_color='#1F2937', node_size=1100, ax=ax)

    other_nodes = [n for n in G_viz.nodes() if n != main_node]
    colors = ['#FFD700' if df_top[df_top['keyword'].apply(fix_persian_text) == node]['status'].values[0] == "طلایی (سبکی)" else '#00C9A7'
              for node in other_nodes]

    nx.draw_networkx_nodes(G_viz, pos, nodelist=other_nodes, node_color=colors, node_size=900, alpha=0.92, edgecolors='#374151', linewidths=1.2, ax=ax)
    nx.draw_networkx_edges(G_viz, pos, alpha=0.35, edge_color='#6B7280', width=1.2, ax=ax)

    for node in other_nodes:
        ax.text(pos[node][0], pos[node][1], node, horizontalalignment='center', verticalalignment='center', fontsize=8, fontweight='bold', color='#111827')

    title_text = fix_persian_text(f"گراف چندلایه کلمات کلیدی - سبک {selected_style}")
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15, color='#111827')
    ax.axis('off')

    df_display = df_top[['keyword', 'status', 'norm_tfidf', 'norm_pr_cooc', 'norm_pr_bert', 'composite_score']].reset_index(drop=True)
    df_display.columns = ['کلمه کلیدی', 'وضعیت سبکی', 'TF-IDF', 'PageRank ساختاری', 'PageRank سیاقی', 'امتیاز نهایی']

    return fig, df_display

# ------------------------------------------------------------------------------
# ۳. رابط کاربری Gradio به همراه بنر راهنما
# ------------------------------------------------------------------------------
custom_css = """
.gradio-container {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
}
.gradio-container * {
    text-align: right !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    border: none !important;
    font-weight: bold !important;
}
"""

with gr.Blocks(title="سامانه تحلیل سبک‌شناختی اشعار (Lite)", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 📜 سامانه تحلیل سبک‌شناختی اشعار فارسی (نسخه Lite & Portable)
        
        این نسخه به عنوان **دموی سریع آنلاین** طراحی شده است تا بدون نیاز به سخت‌افزار سنگین روی وب اجرا شود.
        
        * **روش تحلیل:** ترکیب الگوریتم TF-IDF، تحلیل گراف هم‌آیی (Co-occurrence) و شباهت سیاقی واژگان.
        * **نسخه پیشرفته ParsBERT:** برای اجرای مدل پژوهشی متکی بر ترنسفورمر ParsBERT، از **[دفترچه Google Colab پروژه](https://colab.research.google.com)** استفاده کنید.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            style_input = gr.Dropdown(
                choices=["عراقی", "خراسانی", "هندی", "معاصر"],
                value="عراقی",
                label="🎯 سبک ادبی مورد نظر"
            )
            top_k_input = gr.Slider(
                minimum=5,
                maximum=50,
                value=25,
                step=5,
                label="🔢 تعداد کلمات کلیدی استخراج‌شده"
            )
            text_input = gr.Textbox(
                lines=5,
                label="✍️ تایپ مستقیم متن یا شعر",
                placeholder="شعر خود را اینجا وارد کنید..."
            )
            file_input = gr.File(
                label="📁 یا آپلود فایل متنی (.txt)",
                file_types=[".txt"]
            )
            submit_btn = gr.Button("⚡ اجرای تحلیل سبک‌شناختی", variant="primary")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="🕸️ گراف چندلایه کلمات کلیدی")
            table_output = gr.Dataframe(label="📊 جدول رتبه‌بندی کلمات کلیدی و امتیازات")

    submit_btn.click(
        fn=analyze_style_pipeline,
        inputs=[text_input, file_input, style_input, top_k_input],
        outputs=[plot_output, table_output]
    )

port = int(os.environ.get("PORT", 8000))
demo.launch(server_name="0.0.0.0", server_port=port)
