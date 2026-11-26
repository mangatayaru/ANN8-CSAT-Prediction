import pandas as pd
import streamlit as st
from prediction_ann8 import predict_dataframe, load_artifacts

st.set_page_config(page_title="Shopzilla CSAT — ANN-8", page_icon="📊", layout="wide")
st.title("Shopzilla Customer Satisfaction Prediction")
st.caption("ANN-8 | TensorFlow/Keras | Five CSAT classes | Controlled training resampling")

try:
    _, _, _, metadata = load_artifacts()
    st.success("ANN-8 model and matching preprocessing artifacts loaded.")
except Exception as exc:
    st.error(str(exc))
    st.info("Place the original dataset beside train_ann8.py and run python train_ann8.py.")
    st.stop()

st.info("Post-interaction quality-review model. Probabilities are model estimates, not guarantees.")
single, bulk = st.tabs(["Single customer prediction", "Bulk CSV prediction"])

with single:
    with st.form("single_prediction"):
        left, right = st.columns(2)
        with left:
            channel = st.selectbox("Channel", ["Inbound", "Email", "Chat"])
            category = st.text_input("Category", "Delivery related")
            subcategory = st.text_input("Sub-category", "Delivery Executive")
            remarks = st.text_area("Customer Remarks", "The delivery person was helpful and delivered on time.")
            tenure = st.selectbox("Tenure Bucket", ["0-30 days", "31-60 days", "61-90 days", "90+ days"])
            shift = st.selectbox("Agent Shift", ["Morning", "Afternoon", "Evening", "Night"])
        with right:
            city = st.text_input("Customer City", "Bengaluru")
            product = st.text_input("Product Category", "Electronics")
            price = st.number_input("Item Price", min_value=0.0, value=1500.0)
            handling = st.number_input("Connected Handling Time", min_value=0.0, value=180.0)
            order = st.text_input("Order Date Time", "15-11-2026 10:30:00")
            issue = st.text_input("Issue Reported At", "16-11-2026 14:20:00")
            response = st.text_input("Issue Responded", "16-11-2026 14:35:00")
        submitted = st.form_submit_button("Predict CSAT", type="primary")
    if submitted:
        row = {
            "channel_name": channel, "category": category, "Sub-category": subcategory,
            "Customer Remarks": remarks, "Tenure Bucket": tenure, "Agent Shift": shift,
            "Customer_City": city, "Product_category": product, "Item_price": price,
            "connected_handling_time": handling, "order_date_time": order,
            "Issue_reported at": issue, "issue_responded": response
        }
        try:
            result = predict_dataframe(pd.DataFrame([row])).iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Predicted CSAT", f"{int(result['Predicted_CSAT'])} / 5")
            c2.metric("Highest class probability", f"{result['Prediction_Confidence_Percent']:.2f}%")
            table = pd.DataFrame({"CSAT Score":[1,2,3,4,5],
                "Probability (%)":[result[f"Probability_CSAT_{i}"] for i in range(1,6)]})
            st.subheader("Class probabilities")
            st.dataframe(table.style.format({"Probability (%)":"{:.3f}%"}), hide_index=True, use_container_width=True)
            st.bar_chart(table.set_index("CSAT Score"))
        except Exception as exc:
            st.exception(exc)

with bulk:
    uploaded = st.file_uploader("Upload customer data CSV", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write(f"Rows uploaded: {len(data):,}")
        st.dataframe(data.head(10), use_container_width=True)
        if st.button("Run bulk prediction", type="primary"):
            try:
                output = predict_dataframe(data)
                st.success(f"Completed {len(output):,} predictions.")
                st.dataframe(output.head(100), use_container_width=True)
                st.download_button("Download predictions CSV",
                    output.to_csv(index=False).encode("utf-8"),
                    "shopzilla_ann8_predictions.csv", "text/csv")
            except Exception as exc:
                st.exception(exc)

with st.expander("Model evaluation"):
    st.write("After training, review reports/ann8_metrics.json, reports/ann8_classification_report.json and reports/ann8_confusion_matrix.png.")
