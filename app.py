import streamlit as st
from predictor import predict_disease, get_accuracy, l1, disease

st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="wide")

st.title(" Disease Predictor using Machine Learning")
st.markdown("""
Welcome to the Disease Predictor app! This application uses machine learning algorithms to predict possible diseases based on symptoms you select.
Please select up to 5 symptoms from the list below and choose a model to get a prediction.
""")

st.sidebar.header("About")
st.sidebar.markdown("""
This app demonstrates machine learning in healthcare.
Models used:
- Decision Tree
- Random Forest
- Naive Bayes

Accuracy on test data is displayed for each model.
""")

# Symptom selection
st.subheader("Select Your Symptoms")
selected_symptoms = st.multiselect(
    "Choose symptoms (up to 5):",
    options=l1,
    max_selections=5,
    help="Select the symptoms you're experiencing."
)

# Model selection
model_options = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Naive Bayes": "naive_bayes"
}
selected_model_display = st.selectbox("Choose a Machine Learning Model:", list(model_options.keys()))

selected_model = model_options[selected_model_display]

# Predict button
if st.button("Predict Disease", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        prediction = predict_disease(selected_symptoms, selected_model)
        accuracy = get_accuracy(selected_model)

        st.success(f"**Predicted Disease:** {prediction}")
        st.info(f"**Model Accuracy:** {accuracy:.2%}")

        st.markdown("### Model Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Decision Tree Accuracy", f"{get_accuracy('decision_tree'):.2%}")
        with col2:
            st.metric("Random Forest Accuracy", f"{get_accuracy('random_forest'):.2%}")
        with col3:
            st.metric("Naive Bayes Accuracy", f"{get_accuracy('naive_bayes'):.2%}")

# Footer
# Footer
st.markdown("---")
st.markdown(
    "Developed by **Mohammed Arhaan** | "
    "Computer Science Engineer | "
    "[GitHub](https://github.com/SNE0X) | "
    "[LinkedIn](https://linkedin.com/in/mohammed-arhaan-87742138b)"
)
