from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_pipeline.joblib"
)

CLASS_ORDER = [
    "normal",
    "dos",
    "probe",
    "r2l",
    "u2r",
]


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Система за откриване на кибератаки",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# Model loading
# ============================================================

@st.cache_resource
def load_model():
    """
    Зарежда обучения Random Forest pipeline.
    """
    return joblib.load(MODEL_PATH)


st.title("Система за откриване на кибератаки")

st.write(
    """
    Приложението използва модел Random Forest за
    класифициране на мрежовия трафик в пет категории:
    `normal`, `dos`, `probe`, `r2l` и `u2r`.
    """
)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        f"Моделът не е открит: {MODEL_PATH}"
    )
    st.stop()

except Exception as error:
    st.error(
        f"Моделът не може да бъде зареден: {error}"
    )
    st.stop()


# ============================================================
# File upload
# ============================================================

uploaded_file = st.file_uploader(
    "Качете CSV файл с мрежови записи",
    type=["csv"],
)


if uploaded_file is not None:

    try:
        input_data = pd.read_csv(
            uploaded_file
        )

    except Exception as error:
        st.error(
            f"CSV файлът не може да бъде прочетен: {error}"
        )
        st.stop()

    if input_data.empty:
        st.warning(
            "Каченият файл не съдържа записи."
        )
        st.stop()

    st.subheader("Преглед на входните данни")

    st.dataframe(
        input_data.head(20),
        use_container_width=True,
    )

    st.caption(
        f"Файлът съдържа {len(input_data):,} записа "
        f"и {len(input_data.columns)} колони."
    )


    # ========================================================
    # Input validation
    # ========================================================

    required_columns = list(
        model.feature_names_in_
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in input_data.columns
    ]

    if missing_columns:
        st.error(
            "Липсват задължителни характеристики: "
            + ", ".join(missing_columns)
        )
        st.stop()

    model_input = input_data[
        required_columns
    ].copy()


    # ========================================================
    # Prediction
    # ========================================================

    if st.button(
        "Анализирай мрежовия трафик",
        type="primary",
    ):

        with st.spinner(
            "Мрежовият трафик се анализира..."
        ):
            predictions = model.predict(
                model_input
            )

            probabilities = model.predict_proba(
                model_input
            )

        results = input_data.copy()

        results["predicted_category"] = (
            predictions
        )

        results["prediction_confidence"] = (
            probabilities.max(axis=1)
        ).round(4)

        normal_records = (
            results["predicted_category"]
            == "normal"
        ).sum()

        attack_records = (
            results["predicted_category"]
            != "normal"
        ).sum()

        st.success(
            "Анализът приключи успешно."
        )


        # ====================================================
        # Main indicators
        # ====================================================

        st.subheader("Обобщение")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Общо записи",
            f"{len(results):,}",
        )

        col2.metric(
            "Нормален трафик",
            f"{int(normal_records):,}",
        )

        col3.metric(
            "Потенциални атаки",
            f"{int(attack_records):,}",
        )


        # ====================================================
        # Prediction distribution
        # ====================================================

        st.subheader(
            "Разпределение на прогнозите"
        )

        prediction_counts = (
            results["predicted_category"]
            .value_counts()
            .reindex(
                CLASS_ORDER,
                fill_value=0,
            )
        )

        st.bar_chart(
            prediction_counts
        )


        # ====================================================
        # Evaluation for labelled files
        # ====================================================

        if "attack_category" in results.columns:

            st.subheader(
                "Оценяване на прогнозите"
            )

            evaluation_data = results[
                results["attack_category"]
                .isin(CLASS_ORDER)
            ].copy()

            if evaluation_data.empty:
                st.warning(
                    "Колоната attack_category не съдържа "
                    "валидни категории."
                )

            else:
                actual = evaluation_data[
                    "attack_category"
                ]

                predicted = evaluation_data[
                    "predicted_category"
                ]

                accuracy = accuracy_score(
                    actual,
                    predicted,
                )

                macro_f1 = f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )

                weighted_f1 = f1_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0,
                )

                metric1, metric2, metric3 = (
                    st.columns(3)
                )

                metric1.metric(
                    "Accuracy",
                    f"{accuracy:.2%}",
                )

                metric2.metric(
                    "Macro F1",
                    f"{macro_f1:.4f}",
                )

                metric3.metric(
                    "Weighted F1",
                    f"{weighted_f1:.4f}",
                )


                # ============================================
                # Confusion matrix
                # ============================================

                st.subheader(
                    "Матрица на объркванията"
                )

                matrix = confusion_matrix(
                    actual,
                    predicted,
                    labels=CLASS_ORDER,
                )

                matrix_df = pd.DataFrame(
                    matrix,
                    index=[
                        f"Реален: {label}"
                        for label in CLASS_ORDER
                    ],
                    columns=[
                        f"Предвиден: {label}"
                        for label in CLASS_ORDER
                    ],
                )

                st.dataframe(
                    matrix_df,
                    use_container_width=True,
                )


        # ====================================================
        # All predictions
        # ====================================================

        st.subheader("Резултати")

        st.dataframe(
            results,
            use_container_width=True,
        )


        # ====================================================
        # Detected attacks
        # ====================================================

        st.subheader(
            "Записи, определени като атаки"
        )

        detected_attacks = results[
            results["predicted_category"]
            != "normal"
        ]

        if detected_attacks.empty:
            st.info(
                "Не са открити потенциални атаки."
            )

        else:
            st.warning(
                f"Открити са "
                f"{len(detected_attacks):,} "
                f"потенциални атаки."
            )

            st.dataframe(
                detected_attacks,
                use_container_width=True,
            )


        # ====================================================
        # Download results
        # ====================================================

        csv_results = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Изтеглете резултатите",
            data=csv_results,
            file_name=(
                "network_traffic_predictions.csv"
            ),
            mime="text/csv",
        )

else:
    st.info(
        "Качете CSV файл, за да започнете анализа."
    )