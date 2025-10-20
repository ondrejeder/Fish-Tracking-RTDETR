import gradio as gr
from ultralytics import RTDETR
from PIL import Image
import numpy as np

# Load your trained RT-DETR model
# Make sure 'best.pt' is uploaded to your HuggingFace Space
model = RTDETR('best.pt')

def predict_image(image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run RT-DETR inference on an image
    
    Args:
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Annotated image and detection details
    """
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
    
    # Convert BGR to RGB (OpenCV uses BGR, Gradio expects RGB)
    plotted_img_rgb = plotted_img[..., ::-1]
    
    # Extract detection information
    boxes = result.boxes
    detections_list = []
    
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        
        detections_list.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': bbox
        })
    
    # Format detection text
    if detections_list:
        det_text = f"**Found {len(detections_list)} detection(s):**\n\n"
        for i, det in enumerate(detections_list, 1):
            det_text += f"{i}. **{det['class']}** - Confidence: {det['confidence']:.1%}\n"
    else:
        det_text = "No detections found. Try adjusting the confidence threshold."
    
    return plotted_img_rgb, det_text

# Create Gradio interface
with gr.Blocks(title="Fish & Scale Detector (RT-DETR)") as demo:
    gr.Markdown(
        """
        # 🐟 Fish & Scale Detector
        ### Powered by RT-DETR
        
        Upload an image to detect **fish** and **scales** with high accuracy!
        
        **Model Performance:**
        - Scale Detection: 100% recall, 98.4% precision
        - Overall mAP50: 96.5%
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            
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
            
            predict_btn = gr.Button("🔍 Detect", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detection Results")
            output_text = gr.Markdown(label="Detections")
    
    # Examples (optional - add some sample images if you have them)
    gr.Examples(
        examples=[
            # Add paths to example images here if you want
            # ["example1.jpg", 0.25, 0.45],
        ],
        inputs=[input_image, conf_slider, iou_slider],
    )
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_image,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, output_text]
    )
    
    gr.Markdown(
        """
        ---
        **Model Details:**
        - Architecture: RT-DETR (Real-Time Detection Transformer)
        - Classes: Fish, Scale
        - Training: 150 epochs on 58 images
        - Framework: Ultralytics
        """
    )

if __name__ == "__main__":
    demo.launch()