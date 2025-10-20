import gradio as gr
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import spaces

# Load your trained RT-DETR model
model = RTDETR('best.pt')

@spaces.GPU
def predict_image(image, scale_cm, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run RT-DETR inference on an image and calculate fish lengths
    """
    if image is None:
        return None, "Please upload an image."
    
    if scale_cm is None or scale_cm <= 0:
        return None, "Please enter a valid scale length in cm."
    
    # Run prediction
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False
    )
    
    # Get the first result
    result = results[0]
    
    # Plot results on image
    plotted_img = result.plot()
    
    # Convert BGR to RGB
    plotted_img_rgb = plotted_img[..., ::-1]
    
    # Extract detection information
    boxes = result.boxes
    
    fish_detections = []
    scale_detections = []
    
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        
        # Calculate bounding box dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        detection = {
            'class': class_name,
            'confidence': confidence,
            'bbox': bbox,
            'width': width,
            'height': height
        }
        
        if class_name.lower() == 'fish':
            fish_detections.append(detection)
        elif class_name.lower() == 'scale':
            scale_detections.append(detection)
    
    # Calculate fish lengths
    if not scale_detections:
        return plotted_img_rgb, "⚠️ **Error:** No scale detected in image. Cannot calculate fish lengths."
    
    if not fish_detections:
        return plotted_img_rgb, "ℹ️ **No fish detected in the image.**"
    
    # Use the first scale detection for calibration
    scale_bbox = scale_detections[0]
    # Use the longer dimension of the scale for calibration
    scale_pixels = max(scale_bbox['height'], scale_bbox['width'])
    pixels_per_cm = scale_pixels / scale_cm
    
    # Calculate fish lengths using longer side of bounding box
    fish_lengths = []
    for fish in fish_detections:
        # Use the LONGER side of bounding box as fish length
        length_pixels = max(fish['width'], fish['height'])
        length_cm = length_pixels / pixels_per_cm
        fish_lengths.append(length_cm)
    
    # Sort lengths for better readability
    fish_lengths.sort(reverse=True)
    
    # Calculate statistics
    fish_count = len(fish_lengths)
    min_length = min(fish_lengths)
    max_length = max(fish_lengths)
    avg_length = sum(fish_lengths) / fish_count
    
    # Format output text
    output_text = f"""## 🐟 Analysis Results

**🐠 FISH COUNT:** {fish_count}

**📏 INDIVIDUAL LENGTHS (cm):**
{', '.join([f'{length:.2f}' for length in fish_lengths])}

---

**📊 STATISTICS:**
- **Minimum Length:** {min_length:.2f} cm
- **Maximum Length:** {max_length:.2f} cm
- **Average Length:** {avg_length:.2f} cm

---

**🔧 Detection Details:**
- Scale length used: {scale_cm} cm
- Scale detected: ✓ (confidence: {scale_detections[0]['confidence']:.1%})
- Fish detected: {fish_count}
- Measurement method: Longer bounding box side
- Confidence threshold: {conf_threshold:.0%}
"""
    
    return plotted_img_rgb, output_text

# Create Gradio interface
with gr.Blocks(title="Fish Length Analyzer (RT-DETR)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐟 Fish Length Analyzer
        ### Powered by RT-DETR
        
        Upload an image with fish and a scale reference to automatically measure fish lengths!
        
        **Model Performance:**
        - Scale Detection: 100% recall, 98.4% precision
        - Overall mAP50: 96.5%
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📤 Upload Image")
            
            scale_input = gr.Number(
                label="📏 Scale Length (cm)",
                info="Enter the actual length of the scale object visible in the image",
                value=10.0,
                minimum=0.1,
                maximum=100.0
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Lower = more detections (but more false positives)"
                )
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="IoU Threshold (NMS)",
                    info="Higher = fewer overlapping boxes"
                )
            
            predict_btn = gr.Button("🔍 Analyze & Measure", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="🎯 Detection Results")
            output_text = gr.Markdown(label="📊 Analysis")
    
    gr.Markdown(
        """
        ---
        ### 📋 How to Use:
        1. **Upload an image** containing fish and a scale reference object
        2. **Enter the scale length** (the actual length of your scale object in cm)
        3. Click **"Analyze & Measure"**
        4. View detected fish with their calculated lengths!
        
        ---
        **Model Details:**
        - Architecture: RT-DETR (Real-Time Detection Transformer)
        - Classes: Fish, Scale
        - Training: 150 epochs, 96.5% mAP50
        - Framework: Ultralytics
        """
    )
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_image,
        inputs=[input_image, scale_input, conf_slider, iou_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()