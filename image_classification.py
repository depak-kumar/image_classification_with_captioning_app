# pip install streamlit
# pip install transformers
# pip install torch
# pip install pandas
# pip install pillow
# pip install plotly

import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Title for the app with styling
st.markdown("<h1 style='text-align: center; color: #Ff0000;'>Image Classification and Captioning App</h1>", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload an image for classification and caption generation.
2. The model will predict the top categories, generate a caption, and display a summary.
3. Choose a graph type to view the interactive confidence chart with scores.
4. You can try different image types to see how the model classifies or captions them.
5. The confidence chart is interactive; hover over bars for exact confidence scores.
6. Choose the graph type to visualize the prediction confidence.
""")

# Load the CLIP model for classification and BLIP model for captioning
@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return clip_model, clip_processor, blip_model, blip_processor

clip_model, clip_processor, blip_model, blip_processor = load_models()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Generate caption using BLIP
    st.write("**Generating caption...**")

    def generate_caption(image):
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    caption = generate_caption(img)
    st.markdown(f"<h3 style='color: #Ff0000;'>Generated Caption:</h3> <b>{caption}</b>", unsafe_allow_html=True)

    # Generate predictions using CLIP
    st.write("**Generating predictions...**")

    # Define candidate labels for classification
    candidate_labels = ["a photo of a cat", "a photo of a dog", "a picture of a bird", 
                        "a person", "a car", "nature", "a landscape", "food", "sports"]

    inputs = clip_processor(text=candidate_labels, images=img, return_tensors="pt", padding=True)

    # Perform classification
    with torch.no_grad():
        logits_per_image = clip_model(**inputs).logits_per_image
        probs = logits_per_image.softmax(dim=1).numpy()[0]

    # Display the top predictions with probabilities
    st.write("**Top Predictions:**")
    top_preds = sorted(zip(candidate_labels, probs), key=lambda x: x[1], reverse=True)

    # Display the predictions in a styled format
    for label, prob in top_preds[:5]:
        st.markdown(f"**{label}**: *{prob * 100:.2f}%*")

    # Generate a summary based on top predictions
    st.markdown(f"<h3 style='color: #Ff0000;'>Summary of the image:</h3> <b>The image is most likely a {top_preds[0][0]}, with a confidence of {top_preds[0][1] * 100:.2f}%.</b>", unsafe_allow_html=True)

    # Sidebar option to choose the type of graph
    st.sidebar.write("**Choose the type of confidence chart:**")
    chart_type = st.sidebar.selectbox("Graph Type", ["Bar Chart", "Pie Chart", "Line Chart"])

    labels, scores = zip(*top_preds)

    # Create the selected graph type
    if chart_type == "Bar Chart":
        st.write("**Prediction Confidence Bar Chart:**")
        fig = go.Figure([go.Bar(x=scores, y=labels, orientation='h', marker=dict(color='lightblue'))])
        fig.update_layout(
            title="Confidence Scores for Predicted Labels",
            xaxis_title="Confidence Score",
            yaxis_title="Predicted Labels",
            yaxis={'categoryorder': 'total ascending'},
            template="plotly_dark",
        )
        st.plotly_chart(fig)

    elif chart_type == "Pie Chart":
        st.write("**Prediction Confidence Pie Chart:**")
        fig = px.pie(values=scores, names=labels, title="Confidence Scores for Predicted Labels")
        st.plotly_chart(fig)

    elif chart_type == "Line Chart":
        st.write("**Prediction Confidence Line Chart:**")
        fig = go.Figure([go.Scatter(x=labels, y=scores, mode='lines+markers', marker=dict(color='lightblue'))])
        fig.update_layout(
            title="Confidence Scores for Predicted Labels",
            xaxis_title="Predicted Labels",
            yaxis_title="Confidence Score",
            template="plotly_dark",
        )
        st.plotly_chart(fig)

# Feedback section
st.sidebar.title("Feedback Form")
st.sidebar.write("Please rate the model's performance and leave your feedback.")

# Rating slider (1-5)
rating = st.sidebar.slider("Rate the model's performance (1 = Poor, 5 = Excellent)", 1, 5)

# Text area for feedback/comments
feedback = st.sidebar.text_area("Leave your feedback or suggestions")

# Button to submit feedback
if st.sidebar.button("Submit Feedback"):
    # If the button is pressed, store the feedback
    feedback_data = {
        "Rating": rating,
        "Feedback": feedback
    }

    # Check if the feedback file exists, if not, create a new one
    feedback_file = "/content/drive/MyDrive/feedback.csv"
    if not os.path.exists(feedback_file):
        # Create a new DataFrame and save as a new CSV file
        df = pd.DataFrame([feedback_data])
        df.to_csv(feedback_file, index=False)
        st.sidebar.success("Feedback submitted successfully!")
    else:
        # Append the new feedback to the existing CSV file
        df = pd.read_csv(feedback_file)
        new_df = pd.DataFrame([feedback_data])
        df = pd.concat([df, new_df], ignore_index=True)  # Use concat instead of append
        df.to_csv(feedback_file, index=False)
        st.sidebar.success("Feedback submitted successfully!")
st.subheader("Made By Deepak Kumar")
# # Optional: Display all stored feedback
# if st.sidebar.checkbox("Show stored feedback"):
#     feedback_file = "/content/drive/MyDrive/feedback.csv"
#     if os.path.exists(feedback_file):
#         st.sidebar.write("Here is the stored feedback so far:")
#         df = pd.read_csv(feedback_file)
#         st.sidebar.dataframe(df)
#     else:
#         st.sidebar.write("No feedback available yet.")
