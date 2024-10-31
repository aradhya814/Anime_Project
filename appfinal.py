import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Define the model architecture matching the original trained model
class AnimeModelArchitecture(nn.Module):
    def __init__(self):
        super(AnimeModelArchitecture, self).__init__()
        self.layer = nn.Linear(784, 784)  # Ensure the output matches the checkpoint

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input to 784 dimensions
        x = self.layer(x)
        return x

@st.cache_resource
def load_model():
    model = AnimeModelArchitecture()  # Instantiate the model
    try:
        model.load_state_dict(torch.load('anime_face_full_model.pth', map_location=torch.device('cpu')))
        model.eval()
    except RuntimeError as e:
        st.error(f"Model loading error: {str(e)}")
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match the input size of 28x28
        transforms.Grayscale(),        # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Adjust CLASS_NAMES based on actual number of classes expected
CLASS_NAMES = {i: f"Class {i}" for i in range(784)}  # Adjust this based on your model's output

st.title("Anime Faces Generator")
st.write("Upload an image to classify the anime character type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        try:
            with st.spinner('Processing image...'):
                model = load_model()
                processed_image = process_image(image)
                prediction, confidence = predict(model, processed_image)

            st.success('Processing complete!')
            if prediction in CLASS_NAMES:
                st.write(f"Prediction: {CLASS_NAMES[prediction]}")
                st.write(f"Confidence: {confidence:.2%}")
            else:
                st.error(f"Prediction index {prediction} is out of range. Please check your model output.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.write("Note: This model classifies anime characters into various personality types.")
