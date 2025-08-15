import os
import streamlit as st
from PIL import Image
import numpy as np

# Упрощенная проверка зависимостей
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    pass

# Упрощенная конфигурация страницы для мобильных
st.set_page_config(
    page_title="Кухонный Ассистент",
    layout="centered"  # Лучше для мобильных
)

# Упрощенный заголовок
st.title("🍳 Кухонный Ассистент")
st.write("Загрузите фото продуктов — получите рецепты")

# Основной контейнер
with st.container():
    uploaded = st.file_uploader(
        "📷 Загрузить фото", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Сфотографируйте продукты в холодильнике"
    )

# Параметры в аккордеоне для экономии места
with st.expander("⚙️ Параметры рецептов", expanded=False):
    max_recipes = st.slider("Количество рецептов", 1, 5, 2)
    dietary = st.selectbox("Диета", ["Нет", "Вегетарианская", "Веганская", "Безглютеновая"])
    time_limit = st.selectbox("Время готовки", ["Любое", "до 15 мин", "до 30 мин", "до 60 мин"])

# Обработка загруженного изображения
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)
        
        detected_items = []
        
        # Распознавание объектов
        if TF_AVAILABLE:
            with st.spinner("🔍 Анализирую изображение..."):
                model = MobileNetV2(weights="imagenet")
                img_resized = img.resize((224, 224))
                x = keras_image.img_to_array(img_resized)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)
                decoded = decode_predictions(preds, top=5)[0]
                
                detected_items = [
                    (name.replace("_", " "), float(prob)) 
                    for _, name, prob in decoded
                    if prob > 0.3  # Фильтр по вероятности
                ]
        else:
            st.warning("Функция распознавания изображений недоступна")
        
        # Редактирование списка продуктов
        if detected_items:
            st.subheader("Найдены продукты:")
            final_ingredients = []
            
            for name, prob in detected_items:
                if st.checkbox(
                    f"{name} ({prob:.0%})",
                    value=True,
                    key=f"ing_{name}"
                ):
                    final_ingredients.append(name)
            
            # Ручное добавление
            with st.expander("➕ Добавить другие продукты"):
                manual_input = st.text_input(
                    "Введите через запятую:",
                    placeholder="яйца, молоко, хлеб"
                )
                if manual_input:
                    final_ingredients.extend([
                        x.strip() for x in manual_input.split(",") 
                        if x.strip()
                    ])
            
            if final_ingredients:
                st.success(f"🍎 Ингредиенты: {', '.join(final_ingredients)}")
                
                # Генерация рецептов
                if st.button("🍳 Сгенерировать рецепты", type="primary"):
                    with st.spinner("🧑‍🍳 Готовлю рецепты..."):
                        # Упрощенная локальная генерация
                        recipes = generate_local_recipes(
                            final_ingredients,
                            max_recipes,
                            dietary,
                            time_limit
                        )
                        
                        st.subheader("🍽️ Ваши рецепты")
                        st.write(recipes)
        else:
            st.info("Не удалось распознать продукты. Добавьте их вручную.")
            manual_input = st.text_input("Введите продукты через запятую:")
            if manual_input:
                final_ingredients = [x.strip() for x in manual_input.split(",") if x.strip()]
                
                if st.button("🍳 Сгенерировать рецепты", type="primary"):
                    with st.spinner("🧑‍🍳 Готовлю рецепты..."):
                        recipes = generate_local_recipes(
                            final_ingredients,
                            max_recipes,
                            dietary,
                            time_limit
                        )
                        st.subheader("🍽️ Ваши рецепты")
                        st.write(recipes)
                        
    except Exception as e:
        st.error(f"Ошибка обработки: {str(e)}")
else:
    st.info("📸 Сфотографируйте продукты или загрузите фото")

# Упрощенная локальная генерация рецептов
def generate_local_recipes(ingredients, count, diet, time):
    """Генерация простых рецептов без OpenAI"""
    if not ingredients:
        return "Нет ингредиентов для рецептов"
    
    base = [
        f"### 🥗 Салат из {ingredients[0]} и {ingredients[-1]}\n"
        f"**Время:** 10-15 мин\n**Ингредиенты:** {', '.join(ingredients[:3])}\n"
        "1. Нарежьте все ингредиенты\n2. Смешайте с маслом\n3. Подавайте свежим\n\n",
        
        f"### 🍳 Омлет с {ingredients[0]}\n"
        f"**Время:** 15 мин\n**Ингредиенты:** яйца, {ingredients[0]}\n"
        "1. Взбейте яйца\n2. Обжарьте с ингредиентами\n3. Подавайте горячим\n\n",
        
        f"### 🥪 Бутерброды с {ingredients[0]}\n"
        "**Время:** 5 мин\n**Ингредиенты:** хлеб, {ingredients[0]}\n"
        "1. Намажьте хлеб\n2. Добавьте ингредиенты\n3. Подавайте\n\n"
    ]
    
    return "\n".join(base[:count])