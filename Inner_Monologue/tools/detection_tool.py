"Object detection tool module."

import logging
import os
from typing import Dict, Any, List, Tuple
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T


logger = logging.getLogger(__name__)


class DetectionTool:
    """Tool for detecting objects in the environment using MDETR."""
    
    def __init__(self, environment: Any = None, confidence_threshold: float = 0.4):
        """
        Initialize the detection tool with the given environment.
        
        Args:
            environment: Optional environment object (for compatibility)
            confidence_threshold: Minimum confidence score for detections (default: 0.4)
        """
        self.environment = environment
        self.name = "detection_tool"
        self.description = "Detect objects in the environment and return their labels using MDETR."
        self.confidence_threshold = confidence_threshold
        
        # Model components
        self.model = None
        self.postprocessor = None
        self.tokenizer = None
        self.transform = None
        
        # Initialize the model
        self._load_model()
        
    def _load_model(self):
        """Load the MDETR model and setup the image transform."""
        logger.info("Loading MDETR model structure...")
        
        # Load model structure without pre-trained weights
        self.model, self.postprocessor = torch.hub.load(
            'ashkamath/mdetr:main', 
            'mdetr_resnet101', 
            pretrained=False, 
            return_postprocessor=True,
            trust_repo=True
        )
        
        # Locate and load weights
        local_path = "/root/.cache/torch/hub/checkpoints/pretrained_resnet101_checkpoint.pth"
        url = "https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth"
        
        if os.path.exists(local_path):
            logger.info(f"Loading weights from local cache: {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
        else:
            logger.info("Downloading weights (cache not found)...")
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
        
        # Fix the "Unexpected key" error
        state_dict = checkpoint["model"]
        bad_key = "transformer.text_encoder.embeddings.position_ids"
        
        if bad_key in state_dict:
            logger.info(f"Removing unexpected key: {bad_key}")
            del state_dict[bad_key]
        
        # Load the cleaned weights into the model
        msg = self.model.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully!")
        
        self.model.eval()
        
        # Get tokenizer from the model
        self.tokenizer = self.model.transformer.tokenizer
        
        # Setup image transform
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def detect_objects(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Detect objects in an image based on a text prompt.
        
        Args:
            image: PIL Image to detect objects in
            text_prompt: Text description of objects to detect
            
        Returns:
            Tuple of (bounding_boxes, scores, detected_object_names)
            - bounding_boxes: Tensor of shape [N, 4] with coordinates [xmin, ymin, xmax, ymax]
            - scores: Tensor of shape [N] with confidence scores
            - detected_object_names: List of detected object names
        """
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        # Stage 1: Encode (Get Memory)
        logger.info("Stage 1: Encoding image and text...")
        memory_cache = self.model(img_tensor, [text_prompt], encode_and_save=True)
        
        # Stage 2: Decode (Get Boxes)
        logger.info("Stage 2: Detecting objects...")
        outputs = self.model(img_tensor, [text_prompt], encode_and_save=False, memory_cache=memory_cache)
        
        # Calculate probabilities and filter
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.confidence_threshold
        
        # Rescale bounding boxes
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.postprocessor(outputs, target_sizes)
        
        # Filter results
        result_dict = results[0]
        bboxes_scaled = result_dict['boxes'][keep]
        scores_scaled = result_dict['scores'][keep]
        labels_scaled = result_dict['labels'][keep]
        
        # Decode object names
        tokenized = self.tokenizer(text_prompt, return_tensors="pt")
        input_ids = tokenized.input_ids[0]
        
        detected_objects = [
            self.tokenizer.decode([input_ids[idx]], skip_special_tokens=True).strip() 
            for idx in labels_scaled
        ]
        
        return bboxes_scaled, scores_scaled, detected_objects
    
    def detect_from_url(self, image_url: str, text_prompt: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Detect objects from an image URL.
        
        Args:
            image_url: URL of the image
            text_prompt: Text description of objects to detect
            
        Returns:
            Tuple of (bounding_boxes, scores, detected_object_names)
        """
        image = Image.open(requests.get(image_url, stream=True).raw)
        return self.detect_objects(image, text_prompt)
    
    def detect_from_path(self, image_path: str, text_prompt: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Detect objects from an image file path.
        
        Args:
            image_path: Path to the image file
            text_prompt: Text description of objects to detect
            
        Returns:
            Tuple of (bounding_boxes, scores, detected_object_names)
        """
        image = Image.open(image_path)
        return self.detect_objects(image, text_prompt)
    
    def visualize_detections(self, image: Image.Image, bboxes: torch.Tensor):
        """
        Visualize detection results on an image.
        
        Args:
            image: PIL Image to visualize
            bboxes: Bounding boxes tensor of shape [N, 4]
        """
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        
        for box in bboxes:
            xmin, ymin, xmax, ymax = box.tolist()
            ax.add_patch(plt.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin,
                fill=False, 
                color='red', 
                linewidth=3
            ))
        
        plt.axis('off')
        plt.show()
    
    def get_objects(self) -> Dict[str, Any]:
        """
        Detect objects in the environment and return their labels.
        (Compatibility method for existing interface)
        """
        # This method can be implemented based on your environment's image capture
        if self.environment is None:
            logger.warning("No environment provided. Use detect_objects() with an image instead.")
            return {"objects": []}
        
        # Placeholder - implement based on your environment's API
        logger.warning("get_objects() requires environment integration")
        return {"objects": []}


def main():
    """Test the DetectionTool class."""
    print("=" * 80)
    print("Testing DetectionTool with MDETR")
    print("=" * 80)
    
    # Initialize the detection tool
    print("\n1. Initializing DetectionTool...")
    detector = DetectionTool(confidence_threshold=0.4)
    
    # Test with a sample image
    print("\n2. Loading test image from COCO dataset...")
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    print(f"   Image size: {image.size}")
    
    # Test detection
    print("\n3. Detecting objects...")
    text_prompt = "remote control"
    print(f"   Text prompt: '{text_prompt}'")
    
    bboxes, scores, detected_objects = detector.detect_objects(image, text_prompt)
    
    # Print results
    print("\n4. Detection Results:")
    print(f"   Number of detections: {len(detected_objects)}")
    print(f"   Detected objects: {detected_objects}")
    print(f"   Confidence scores: {scores.tolist()}")
    print(f"   Bounding boxes shape: {bboxes.shape}")
    
    # Print detailed results
    print("\n5. Detailed Results:")
    for i, (obj_name, score, bbox) in enumerate(zip(detected_objects, scores, bboxes)):
        xmin, ymin, xmax, ymax = bbox.tolist()
        print(f"   Detection {i+1}:")
        print(f"      Object: {obj_name}")
        print(f"      Score: {score:.4f}")
        print(f"      BBox: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")
    
    # Visualize results
    print("\n6. Visualizing results...")
    detector.visualize_detections(image, bboxes)
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()