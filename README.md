# image_classification_with_captioning_app
![image](https://github.com/user-attachments/assets/09b72eb4-8837-4635-9699-7e7944d2afbe)
This code is a Streamlit-based web app that provides image classification and captioning functionalities using CLIP and BLIP models. The application features interactive elements like an image uploader, prediction confidence graphs, and a feedback form for user input. Key components include:

Title and Sidebar Instructions:

The app's title is displayed in red with centered alignment.
The sidebar provides step-by-step instructions for using the app, which include uploading an image, generating predictions, and interacting with confidence charts.
Model Loading:

The app uses @st.cache_resource to load the CLIP and BLIP models efficiently, allowing them to be cached and reducing load times when re-running.
Image Upload and Display:

Users can upload images in JPEG or PNG format.
The uploaded image is displayed within the app for quick preview.
Caption Generation:

The app generates a caption for the uploaded image using the BLIP model.
A function generate_caption preprocesses the image, runs it through the model, and decodes the result to display a descriptive caption.
Image Classification:

The CLIP model classifies the image based on predefined candidate labels, which include categories like "a photo of a cat" or "a picture of a bird."
The app sorts and displays the top predictions with their confidence scores.
Prediction Confidence Summary:


![image](https://github.com/user-attachments/assets/298d8bf6-62dc-451e-b5ae-75b1f1d3c345)
![image](https://github.com/user-attachments/assets/8bfc0a1b-4a20-4b0b-861e-4903e5546d8c)
The highest confidence prediction is summarized in bold text to give users a quick overview of the model's output.
Confidence Visualization Options:

Users can choose between Bar Chart, Pie Chart, and Line Chart to visualize prediction confidence scores interactively using Plotly.
Each graph type is customized for readability and visual appeal.
Feedback Section:

A feedback form in the sidebar allows users to rate the model’s performance on a scale of 1 to 5 and leave comments.
Upon submission, feedback is stored in a CSV file, with new feedback either creating a new file or appending to an existing one.
Developer Credit:

The developer’s name is displayed at the end of the app, adding a personalized touch.
This app is an effective demonstration of computer vision capabilities, offering both classification and captioning within an intuitive, user-friendly interface. The additional feedback section encourages user interaction and iterative improvement based on real user input.
![image](https://github.com/user-attachments/assets/8a51b81b-437c-49e8-b8a7-1baf3000fdda)
![image](https://github.com/user-attachments/assets/6c51bd20-48e5-43c9-8bca-823e3a4ba777)
![image](https://github.com/user-attachments/assets/add4b156-b51b-4a97-a5d8-fabfc7a79597)
