import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path, num_classes):
    # Initialize the model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_font_list(font_list_path):
    with open(font_list_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_font(model, image_tensor, font_list, top_k=5):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'font': font_list[idx],
                'probability': prob.item()
            })
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict font from an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='models/best_resnet50.pth',
                      help='Path to the trained model weights')
    parser.add_argument('--font_list', type=str, default='fontlist.txt',
                      help='Path to the font list file')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        return
    
    # Load model and font list
    try:
        font_list = load_font_list(args.font_list)
        model = load_model(args.model_path, len(font_list))
    except Exception as e:
        print(f"Error loading model or font list: {str(e)}")
        return
    
    # Process image and make prediction
    try:
        image_tensor = preprocess_image(args.image_path)
        predictions = predict_font(model, image_tensor, font_list, args.top_k)
        
        print("\nTop font predictions:")
        print("-" * 50)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['font']}: {pred['probability']*100:.2f}%")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    main() 