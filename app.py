
# app.py
import os
import streamlit as st
from PIL import Image
import numpy as np
import io

# Optional: use tensorflow / keras for offline image classification
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional OpenAI integration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = OPENAI_API_KEY is not None
if USE_OPENAI:
    import openai
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="AI Кухонный Ассистент — Demo", layout="wide")
st.title("AI Кухонный Ассистент — прототип")
st.markdown("Загрузите фото с продуктами из холодильника — приложение распознает продукты и предложит рецепты.")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Загрузить фото (или несколько)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    st.write("Параметры:")
    max_recipes = st.slider("Сколько рецептов сгенерировать", 1, 5, 3)
    dietary = st.selectbox("Диета / ограничение", ["Нет", "Вегетарианская", "Веганская", "Безглютеновая", "Низкоуглеводная"])
    time_limit = st.selectbox("Максимальное время приготовления", ["Любое", "до 15 мин", "до 30 мин", "до 60 мин"])

with col2:
    st.write("Результат:")

if uploaded is None:
    st.info("Загрузите фото для начала (например, полка холодильника).")
else:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Загруженное фото", use_column_width=True)

    # --- Step 1: Simple CV detection using MobileNetV2 (ImageNet labels)
    detected_items = []
    if TF_AVAILABLE:
        st.info("Распознавание: модель MobileNetV2 (ImageNet) — результаты предварительные.")
        model = MobileNetV2(weights="imagenet")
        # prepare image
        img_resized = img.resize((224,224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded = decode_predictions(preds, top=6)[0]  # list of tuples (class, name, prob)
        for _, name, prob in decoded:
            detected_items.append((name.replace("_", " "), float(prob)))
    else:
        st.warning("TensorFlow не установлен — использую 'mock' распознавание. Укажи вручную продукты.")
        # fallback: let user input items manually
        manual = st.text_input("Вручную перечислите продукты через запятую (напр. tomato, milk, eggs):")
        if manual:
            for token in [t.strip() for t in manual.split(",") if t.strip()]:
                detected_items.append((token, 0.6))

    if detected_items:
        st.subheader("Предварительно распознано (можно отредактировать):")
        # Show checkboxes and allow editing/confirm
        cols = st.columns([3,1])
        with cols[0]:
            edited = []
            for name, prob in detected_items:
                new_name = st.text_input(f"Продукт (вероятность {prob:.2f})", value=name, key=name+str(prob))
                edited.append(new_name.strip())
        with cols[1]:
            # Allow removing
            st.write("")
        # Let user confirm or add
        more = st.text_input("Добавить ещё продукты вручную (через запятую):", "")
        if more:
            for t in [x.strip() for x in more.split(",") if x.strip()]:
                edited.append(t)
        # final list
        final_ingredients = [x for x in edited if x]
        st.write("Итоговый список ингредиентов:", final_ingredients)

        # --- Step 2: Generate recipes using OpenAI (если есть) либо локальный шаблон
        if st.button("Сгенерировать рецепты"):
            with st.spinner("Генерирую рецепты..."):
                prompt = f"""У меня есть следующие ингредиенты: {', '.join(final_ingredients)}.
Сгенерируй {max_recipes} рецептов, коротко опиши шаги приготовления, время, примерную калорийность (примерно), 
укажи какие дополнительные ингредиенты могут понадобиться, и предложи вариант под ограничение: {dietary}. 
Если времени ограничения: {time_limit}.
Формат: заголовок рецепта, время, ингредиенты (из холодильника пометь), дополнительные ингредиенты, шаги, калории примерно."""
                recipes_text = ""
                if USE_OPENAI:
                    try:
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",  # пример, заменить на доступный модельный ID
                            messages=[{"role":"system","content":"Ты — помощник‑кухня. Пиши компактно и полезно."},
                                      {"role":"user","content":prompt}],
                            max_tokens=800,
                            temperature=0.8,
                        )
                        recipes_text = resp['choices'][0]['message']['content']
                    except Exception as e:
                        st.error("Ошибка при обращении к OpenAI: " + str(e))
                        recipes_text = local_recipe_generator(final_ingredients, max_recipes, dietary, time_limit)
                else:
                    recipes_text = local_recipe_generator(final_ingredients, max_recipes, dietary, time_limit)

                st.markdown("### Результат генерации")
                st.text_area("Рецепты", value=recipes_text, height=400)
    else:
        st.info("Нет распознанных продуктов — подправьте вручную.")

# --- Local fallback recipe generator
def local_recipe_generator(ingredients, max_recipes, dietary, time_limit):
    """
    Простой локальный генератор рецептов (шаблоны), чтобы демо работал без OpenAI.
    """
    if not ingredients:
        return "Нет ингредиентов для генерации."

    templates = [
        ("Омлет с дополнениями", """
Время: ~15 минут
Ингредиенты: яйца (2-3), {extras}
Доп. ингредиенты: соль, перец, масло
Шаги:
1. Взбей яйца с солью и перцем.
2. На сковороде разогреть масло, добавить {extras} (измельчённые) и обжарить 2-3 мин.
3. Вылить яйца, готовить до готовности.
Примерные калории: 300-450 ккал
"""),
        ("Овощной салат быстрого приготовления", """
Время: ~10 минут
Ингредиенты: {veggies}
Доп. ингредиенты: оливковое масло, лимон/уксус, соль
Шаги:
1. Нарежьте ингредиенты мелко.
2. Смешайте с маслами и приправами.
3. Подайте холодным.
Калории: ~150-350 ккал
"""),
        ("Паста/рис с овощами", """
Время: ~20-30 минут
Ингредиенты: паста/рис, {mix}
Доп. ингредиенты: соль, перец, сыр (по вкусу)
Шаги:
1. Отварить пасту/рис.
2. Обжарить {mix} на сковороде.
3. Смешать и подать.
Калории: ~400-700 ккал
""")
    ]

    chosen = []
    ing_str = ", ".join(ingredients)
    veggies = ", ".join([i for i in ingredients if any(k in i.lower() for k in ["tomato","pepper","cucumber","lettuce","onion","carrot","salad","vegetable","potato"])][:3]) or "овощи"
    extras = ", ".join(ingredients[:2])
    mix = ", ".join(ingredients[:3]) or "ингредиенты"

    for i, t in enumerate(templates[:max_recipes]):
        title, body = t
        body_f = body.format(extras=extras, veggies=veggies, mix=mix)
        chosen.append(f"---\n{title}\n{body_f}\n")
    return "\n".join(chosen)




