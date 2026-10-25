import streamlit as st

st.set_page_config(page_title="Data Analysis Projects", layout="wide")

lab2 = st.Page("pages/1_Lab2_EDA.py", title="Lab2 EDA")
lab3 = st.Page("pages/2_Lab3_Regression.py", title="Lab3 Regression")
lab4 = st.Page("pages/3_Lab4_Classification.py", title="Lab4 Classification")


def home():
    st.markdown(
        """
        <style>
        a[data-testid="stPageLink-NavLink"] p {
            font-size: 1.75rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Data Analysis Projects")
    st.subheader("Технологии анализа и визуализации данных")
    st.caption("Автор: Palina Bohush")
    st.markdown("[GitHub: Data-Analysis-Projects](https://github.com/bohushpalina/Data-Analysis-Projects)")

    st.write(
        "Веб-приложение содержит интерактивные дополнительные задания "
        "лабораторных работ 2–4. Для перехода к нужной работе используйте меню в левой части страницы."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.page_link(lab2, label="Лабораторная 2")
        st.write(
            "Разведочный анализ данных. Выбор датасета, очистка данных "
            "(дубликаты, пропуски), фильтрация, динамические таблицы и выбор типа графика."
        )

    with col2:
        st.page_link(lab3, label="Лабораторная 3")
        st.write(
            "Линейная регрессия. Ввод значений признаков с помощью ползунков, "
            "расчёт прогноза по обученной модели и интерактивный график предсказаний."
        )

    with col3:
        st.page_link(lab4, label="Лабораторная 4")
        st.write(
            "Классификация. Выбор классификатора, настройка гиперпараметров, "
            "матрица неточностей и таблица метрик качества."
        )

    st.info(
        "При первом открытии лабораторных работ 2 и 4 требуется время на загрузку датасетов."
    )


home_page = st.Page(home, title="Data Analysis Projects", default=True)

st.navigation([home_page, lab2, lab3, lab4]).run()
