import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("🧠 Classification Model Builder")
st.markdown("Upload a CSV file to build a Machine Learning Model on it and predict the outcome as quick view")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded data with {df.shape[0]:,} rows and {df.shape[1]} columns.")
        with st.expander("🧾 Basic Info"):
            st.write("**Shape:**", df.shape)
            st.write("**Column Names:**", list(df.columns))
        with st.expander("👀 Preview Data"):
            if st.checkbox("Show first 5 rows"):
                st.dataframe(df.head())
            if st.checkbox("Show last 5 rows"):
                st.dataframe(df.tail())
        with st.expander("📊 Summary Statistics"):
            st.dataframe(df.describe())

        # Drop Irrelevant Columns
        drop_cols = st.multiselect(
            "Select columns to drop that are not required for Model",
            options=list(df.columns),
            default=[]
        )
        df = df.drop(columns=drop_cols)

        # Target selection
        target = st.selectbox("🎯 Select the target column", options=[None] + list(df.columns), index=0)
        if target:
            X = df.drop(columns=[target])
            y = df[target]

            # Encode categorical variables
            X = pd.get_dummies(X)
            st.write("🧪 Features after encoding:")
            st.dataframe(X.head())
            
            # Train/Test Split
            test_size = st.slider('Test Set Size (%)', 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            st.success(f"Data Splitted succesfully. Shape of X Train : {X_train.shape} and X Test : {X_test.shape}")

            # Choose Model
            model_choice = st.selectbox("🛠️ Choose model",
                [None, "Random Forest", "Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Gradient Boosting"]
            )
            if model_choice is not None:
                col1, col2 = st.columns(2)
                if model_choice == 'Random Forest':
                    with col1:
                        n_estimators = st.slider("Number of Trees", 10, 200, 100, step=10)
                    with col2:
                        max_depth = st.selectbox("Max Tree Depth", [None] + list(range(2, 21)))
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                elif model_choice == 'Logistic Regression':
                    with col1:
                        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
                    with col2:
                        max_iter = st.slider("Max Iteration", 100, 2000, 500, step=100)
                    model = LogisticRegression(C=C, max_iter=max_iter)
                elif model_choice == 'K-Nearest Neighbors (KNN)':
                    with col1:
                        n_neighbors = st.slider("Number of Neighbors", 1, 15, 5)
                    with col2:
                        weights = st.selectbox("Weighting Scheme", ["uniform", "distance"])
                    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                elif model_choice == 'Support Vector Machine (SVM)':
                    with col1:
                        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
                    with col2:
                        kernel = st.selectbox("Kernel Type", ["linear", "rbf"])
                    model = SVC(C=C, kernel=kernel)
                elif model_choice == 'Gradient Boosting':
                    with col1:
                        n_estimators = st.slider("Number of Trees", 10, 200, 100, step=10)
                    with col2:
                        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

                # Model Training and Prediction
                if st.button("🚀 Train Model"):
                    model.fit(X_train, y_train)
                    st.success('✅ Model trained successfully!')
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{acc:.2f}")

                    cm = confusion_matrix(y_test, y_pred)
                    labels = model.classes_
                    cm_df = pd.DataFrame(cm,
                        index=[f"Actual {label}" for label in labels],
                        columns=[f"Predicted {label}" for label in labels]                     
                    )
                    # Displaying Confusion Matrix
                    st.subheader("📊 Confusion Matrix")
                    st.dataframe(cm_df)
    except Exception as error:
        print(error)