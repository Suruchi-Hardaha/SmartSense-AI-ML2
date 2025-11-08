
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import argparse
import json
import os

# Category mapping (must match training)
CATEGORY_MAP = {
    1: 'bathroom',
    2: 'bedroom',
    3: 'garage',
    4: 'hall',
    5: 'kitchen',
    6: 'laundry',
    7: 'porch',
    8: 'room'
}

def create_model(num_classes=9, weights_path="floorplan_model_weights.pth", device='cpu'):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def parse_floorplan(image_path, model, device='cpu', conf_thresh=0.6):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs['boxes']
    labels = outputs['labels']
    scores = outputs['scores']

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
        detections.append({
            "label": CATEGORY_MAP[int(label)],
            "score": float(score),
            "bbox": box.cpu().numpy().tolist()
        })

    # Count per class
    summary = {}
    for det in detections:
        label = det["label"]
        summary[label] = summary.get(label, 0) + 1

    result = {
        "rooms": summary.get("room", 0),
        "halls": summary.get("hall", 0),
        "kitchens": summary.get("kitchen", 0),
        "bathrooms": summary.get("bathroom", 0),
        "rooms_detail": [
            {"label": label.title(), "count": count, "approx_area": None}
            for label, count in summary.items()
        ]
    }

    return result, detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a floorplan image into structured JSON")
    parser.add_argument("--image", type=str, required=True, help="Path to the floorplan image")
    parser.add_argument("--weights", type=str, default="floorplan_model_weights.pth", help="Path to model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(weights_path=args.weights, device=device)
    result, detections = parse_floorplan(args.image, model, device)

    print(json.dumps(result, indent=2))
