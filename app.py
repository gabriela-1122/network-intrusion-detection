from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "network_traffic_sample.csv"

CLASS_ORDER = ["normal", "dos", "probe", "r2l", "u2r"]
CLASS_DESCRIPTIONS = {
    "normal": "Нормална мрежова активност",
    "dos": "Отказ от услуга",
    "probe": "Проучване на хостове и услуги",
    "r2l": "Отдалечен достъп до локален акаунт",
    "u2r": "Получаване на администраторски права",
}

CLASS_ICONS = {
    "normal": "✅",
    "dos": "⛔",
    "probe": "🔎",
    "r2l": "🔐",
    "u2r": "⚠️",
}

CLASS_TITLES = {
    "normal": "Нормален трафик",
    "dos": "Отказ от услуга",
    "probe": "Проучване на мрежата",
    "r2l": "Remote-to-Local",
    "u2r": "User-to-Root",
}



st.set_page_config(
    page_title="Система за откриване на кибератаки",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .attack-card {
        min-height: 178px;
        padding: 22px 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(49, 51, 63, 0.20);
        border-radius: 12px;
        background: var(--secondary-background-color);
        box-sizing: border-box;
    }

    .attack-card-code {
        margin-bottom: 24px;
        color: #858892;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
    }

    .attack-card-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
        color: var(--text-color);
        font-size: clamp(1.15rem, 1.55vw, 1.55rem);
        font-weight: 700;
        line-height: 1.2;
        white-space: nowrap;
    }

    .attack-card-icon {
        flex: 0 0 auto;
        font-size: 1.75rem;
        line-height: 1;
    }

    .attack-card-description {
        color: var(--text-color);
        font-size: 1rem;
        line-height: 1.45;
    }

    @media (max-width: 1100px) {
        .attack-card-title {
            white-space: normal;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Зарежда обучения Random Forest pipeline."""
    return joblib.load(MODEL_PATH)


def get_model():
    """Зарежда модела и показва разбираема грешка при проблем."""
    try:
        return load_model()
    except FileNotFoundError:
        st.error(f"Моделът не е открит: {MODEL_PATH}")
    except Exception as error:
        st.error(f"Моделът не може да бъде зареден: {error}")
    st.stop()


def show_header(title, description):
    st.title(title)
    st.write(description)

def show_attack_card(container, class_name):
    """Показва четима карта с пълно описание на категорията."""
    with container:
        st.markdown(
            f"""
            <div class="attack-card">
                <div class="attack-card-code">{class_name.upper()}</div>
                <div class="attack-card-title">
                    <span class="attack-card-icon">{CLASS_ICONS[class_name]}</span>
                    <span>{CLASS_TITLES[class_name]}</span>
                </div>
                <div class="attack-card-description">
                    {CLASS_DESCRIPTIONS[class_name]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def show_home_page():
    show_header(
        "🛡️ Система за откриване на кибератаки",
        "Интерактивен прототип за многокласова класификация на мрежов трафик.",
    )

    st.info(
        "Системата използва обучен Random Forest модел и класифицира всеки "
        "мрежов запис в една от пет категории."
    )

    st.subheader("Категории на мрежовия трафик")

    first_row = st.columns(3)
    for column, class_name in zip(first_row, CLASS_ORDER[:3]):
        show_attack_card(column, class_name)

    second_row = st.columns(2)
    for column, class_name in zip(second_row, CLASS_ORDER[3:]):
        show_attack_card(column, class_name)

    st.subheader("Как работи приложението")
    step1, step2, step3 = st.columns(3)
    step1.markdown("### 1. Качване\nКачете CSV файл с необходимите 39 входни характеристики.")
    step2.markdown("### 2. Анализ\nМоделът обработва записите и определя тяхната категория.")
    step3.markdown("### 3. Резултати\nПрегледайте прогнозите и изтеглете обогатения CSV файл.")

    st.subheader("Важно ограничение")
    st.warning(
        "Приложението е учебен и изследователски прототип. То работи с предварително "
        "извлечени характеристики от KDD Cup 1999 и не приема директно PCAP файлове "
        "или мрежов поток в реално време."
    )

    if SAMPLE_PATH.exists():
        st.subheader("Демонстрационни данни")
        st.write("Можете да изтеглите примерния файл и да го качите в раздел „Анализ на трафика“.")
        st.download_button(
            "Изтеглете демонстрационния CSV",
            data=SAMPLE_PATH.read_bytes(),
            file_name=SAMPLE_PATH.name,
            mime="text/csv",
        )


def create_results(input_data, model):
    required_columns = list(model.feature_names_in_)
    missing_columns = [column for column in required_columns if column not in input_data.columns]

    if missing_columns:
        st.error("Липсват задължителни характеристики: " + ", ".join(missing_columns))
        return None

    model_input = input_data[required_columns].copy()

    try:
        predictions = model.predict(model_input)
        probabilities = model.predict_proba(model_input)
    except Exception as error:
        st.error(f"Данните не могат да бъдат обработени от модела: {error}")
        return None

    results = input_data.copy()
    results["predicted_category"] = predictions
    results["prediction_confidence"] = probabilities.max(axis=1).round(4)
    return results


def show_evaluation(results):
    if "attack_category" not in results.columns:
        st.info(
            "Файлът няма колона attack_category. Прогнозите са създадени, "
            "но оценъчни метрики не могат да бъдат изчислени."
        )
        return

    evaluation_data = results[results["attack_category"].isin(CLASS_ORDER)].copy()
    invalid_count = len(results) - len(evaluation_data)

    if evaluation_data.empty:
        st.warning("Колоната attack_category не съдържа валидни категории.")
        return

    if invalid_count:
        st.warning(f"{invalid_count:,} записа са изключени от оценяването поради невалидна категория.")

    actual = evaluation_data["attack_category"]
    predicted = evaluation_data["predicted_category"]

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    weighted_f1 = f1_score(actual, predicted, average="weighted", zero_division=0)

    st.subheader("Оценяване на прогнозите")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Accuracy", f"{accuracy:.2%}")
    metric2.metric("Macro F1", f"{macro_f1:.4f}")
    metric3.metric("Weighted F1", f"{weighted_f1:.4f}")

    st.subheader("Матрица на объркванията")
    matrix = confusion_matrix(actual, predicted, labels=CLASS_ORDER)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Реален: {label}" for label in CLASS_ORDER],
        columns=[f"Предвиден: {label}" for label in CLASS_ORDER],
    )
    st.dataframe(matrix_df, use_container_width=True)


def show_results(results):
    normal_records = int((results["predicted_category"] == "normal").sum())
    attack_records = int((results["predicted_category"] != "normal").sum())

    st.success("Анализът приключи успешно.")
    st.subheader("Обобщение")
    col1, col2, col3 = st.columns(3)
    col1.metric("Общо записи", f"{len(results):,}")
    col2.metric("Нормален трафик", f"{normal_records:,}")
    col3.metric("Потенциални атаки", f"{attack_records:,}")

    st.subheader("Разпределение на прогнозите")
    prediction_counts = (
        results["predicted_category"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    )
    st.bar_chart(prediction_counts)

    show_evaluation(results)

    st.subheader("Всички резултати")
    st.dataframe(results, use_container_width=True)

    st.subheader("Записи, определени като атаки")
    detected_attacks = results[results["predicted_category"] != "normal"]
    if detected_attacks.empty:
        st.info("Не са открити потенциални атаки.")
    else:
        st.warning(f"Открити са {len(detected_attacks):,} потенциални атаки.")
        st.dataframe(detected_attacks, use_container_width=True)

    csv_results = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Изтеглете резултатите",
        data=csv_results,
        file_name="network_traffic_predictions.csv",
        mime="text/csv",
    )


def show_analysis_page():
    show_header(
        "🔍 Анализ на мрежовия трафик",
        "Качете структуриран CSV файл, за да класифицирате мрежовите записи.",
    )
    model = get_model()
    required_columns = list(model.feature_names_in_)

    with st.expander("Изисквания към входния файл"):
        st.write(
            "Файлът трябва да съдържа следните 39 входни характеристики. "
            "Допълнителните колони attack_type и attack_category са позволени."
        )
        st.code(", ".join(required_columns), language=None)

    uploaded_file = st.file_uploader(
        "Качете CSV файл с мрежови записи",
        type=["csv"],
        key="traffic_file",
    )

    if uploaded_file is None:
        st.info("Качете CSV файл, за да започнете анализа.")
        return

    try:
        input_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"CSV файлът не може да бъде прочетен: {error}")
        return

    if input_data.empty:
        st.warning("Каченият файл не съдържа записи.")
        return

    st.subheader("Преглед на входните данни")
    st.dataframe(input_data.head(20), use_container_width=True)
    st.caption(f"Файлът съдържа {len(input_data):,} записа и {len(input_data.columns)} колони.")

    if st.button("Анализирай мрежовия трафик", type="primary"):
        with st.spinner("Мрежовият трафик се анализира..."):
            results = create_results(input_data, model)
        if results is not None:
            st.session_state["analysis_results"] = results

    if "analysis_results" in st.session_state:
        show_results(st.session_state["analysis_results"])


def show_model_page():
    show_header(
        "🌲 Информация за модела",
        "Обобщение на използвания набор, крайния модел и експерименталните резултати.",
    )

    st.subheader("Краен модел")
    left, right = st.columns(2)
    left.markdown(
        """
        - **Алгоритъм:** Random Forest
        - **Брой дървета:** 200
        - **Случайна начална стойност:** 42
        - **Входни характеристики:** 39
        - **Категории:** normal, dos, probe, r2l, u2r
        """
    )
    right.markdown(
        """
        - **Training + validation:** 145 586 уникални записа
        - **Corrected test set:** 311 029 записа
        - **Предварителна обработка:** общ scikit-learn pipeline
        - **Категориални полета:** protocol_type, service, flag
        """
    )

    st.subheader("Резултати")
    result1, result2, result3 = st.columns(3)
    result1.metric("Validation Macro F1", "0.9602")
    result2.metric("Test Accuracy", "0.9261")
    result3.metric("Test Macro F1", "0.6023")

    metrics = pd.DataFrame(
        {
            "Клас": ["Normal", "DoS", "Probe", "R2L", "U2R"],
            "Precision": [0.7286, 0.9989, 0.9223, 0.9914, 0.6875],
            "Recall": [0.9953, 0.9734, 0.7609, 0.0496, 0.1571],
            "F1": [0.8413, 0.9860, 0.8339, 0.0945, 0.2558],
        }
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Интерпретация")
    st.write(
        "Моделът разпознава надеждно Normal, DoS и Probe, но има ниска чувствителност "
        "при R2L и U2R. Той класифицира правилно 98.10% от записите с познати типове "
        "атаки и 6.82% от записите с непознати типове."
    )
    st.warning(
        "Получените резултати не доказват готовност за директно внедряване в реална "
        "организация, защото KDD Cup 1999 е исторически набор."
    )


with st.sidebar:
    st.title("Навигация")
    selected_page = st.radio(
        "Изберете раздел",
        ["Начало", "Анализ на трафика", "Информация за модела"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Дипломен проект: класификация на кибератаки")


if selected_page == "Начало":
    show_home_page()
elif selected_page == "Анализ на трафика":
    show_analysis_page()
else:
    show_model_page()
