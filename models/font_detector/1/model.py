import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import triton_python_backend_utils as pb_utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "Deepfont.pth")
        font_list_path = os.path.join(model_dir, "fontsubset.txt")
        
        # Check if required files exist
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found at {model_path}")
        if not os.path.exists(font_list_path):
            raise RuntimeError(f"Font list file not found at {font_list_path}")
        
        # Get device info from args
        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using GPU {device_id}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")

        # Load model
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Load font classes
        try:
            with open(font_list_path, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.classes)} font classes")
        except Exception as e:
            raise RuntimeError(f"Failed to load font classes: {str(e)}")

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image_data):
        try:
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)

            if isinstance(image_data, bytes):
                image_data = image_data.decode("utf-8")
                image_data = base64.b64decode(image_data)

            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0)
            return img_tensor
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise

    def execute(self, requests):
        try:
            # Gather inputs from all requests
            batched_inputs = []
            for request in requests:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
                input_data_array = in_tensor.as_numpy()
                batched_inputs.append(self.preprocess(input_data_array[0, 0]))
            
            # Combine inputs along the batch dimension
            batched_tensor = torch.cat(batched_inputs, dim=0).to(self.device)
            logger.debug(f"Processing batch of size {len(batched_inputs)}")
            
            # Run inference once on the full batch
            with torch.no_grad():
                outputs = self.model(batched_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, 1)
                confidences = probabilities[torch.arange(len(probabilities)), predicted_classes]
            
            # Process the outputs and split them for each request
            responses = []
            for i, request in enumerate(requests):
                predicted_label = self.classes[predicted_classes[i].item()]
                probability = confidences[i].item()
                
                # Create numpy arrays with shape [1, 1] for consistency
                out_label_np = np.array([[predicted_label]], dtype=object)
                out_prob_np = np.array([[probability]], dtype=np.float32)
                
                out_tensor_label = pb_utils.Tensor("FONT_LABEL", out_label_np)
                out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_label, out_tensor_prob])
                responses.append(inference_response)
            
            return responses
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            raise 