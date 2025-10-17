"""
Fish Counter & Measurement Web App
This is your Gradio interface for detecting and measuring fish
"""

import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import os

# Check if model exists
MODEL_PATH = 'best.pt'

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ WARNING: Model file '{MODEL_PATH}' not found!")
    print("Please copy your trained model (best.pt) to the project folder.")
    model = None
else:
    print(f"✅ Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")

def analyze_fish(image, scale_length_cm):
    """
    Main function that processes uploaded image
    
    Args:
        image: numpy array of uploaded image
        scale_length_cm: length of the scale/ruler in centimeters
    
    Returns:
        annotated_image: image with bounding boxes drawn
        results_text: formatted text with fish count and measurements
    """
    
    if model is None:
        return None, "❌ ERROR: Model not loaded. Please ensure best.pt exists in the project folder."
    
    if image is None:
        return None, "❌ Please upload an image first!"
    
    # Run YOLO detection
    results = model(image, conf=0.25)[0]  # conf=0.25 is confidence threshold
    
    # Parse detections
    fish_detections = []
    scale_detection = None
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        confidence = float(box.conf[0])
        
        if class_name == 'fish':
            fish_detections.append({
                'bbox': bbox,
                'confidence': confidence
            })
        elif class_name == 'scale':
            scale_detection = bbox
    
    # Get annotated image (YOLO draws boxes automatically)
    annotated_image = results.plot()
    
    # Convert BGR to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Calculate measurements if scale is detected
    if scale_detection is not None and len(fish_detections) > 0:
        # Calculate pixels per cm from scale
        x1, y1, x2, y2 = scale_detection
        scale_width = abs(x2 - x1)
        scale_height = abs(y2 - y1)
        scale_pixels = max(scale_width, scale_height)  # Use longer dimension
        
        pixels_per_cm = scale_pixels / scale_length_cm
        
        # Measure each fish
        fish_lengths = []
        for i, fish in enumerate(fish_detections):
            x1, y1, x2, y2 = fish['bbox']
            fish_width = abs(x2 - x1)
            fish_height = abs(y2 - y1)
            fish_pixels = max(fish_width, fish_height)  # Use longer dimension
            
            length_cm = fish_pixels / pixels_per_cm
            fish_lengths.append(round(length_cm, 2))
        
        # Create results text
        results_text = f"""
🐟 **FISH COUNT:** {len(fish_detections)}

📏 **INDIVIDUAL LENGTHS (cm):**
{', '.join(map(str, fish_lengths))}

📊 **STATISTICS:**
- Average Length: {np.mean(fish_lengths):.2f} cm
- Minimum Length: {min(fish_lengths):.2f} cm
- Maximum Length: {max(fish_lengths):.2f} cm

✅ **Scale Detected:** Yes ({scale_length_cm} cm)
⚙️ **Scale Factor:** {pixels_per_cm:.2f} pixels/cm
        """
    
    elif scale_detection is None and len(fish_detections) > 0:
        # Fish detected but no scale
        results_text = f"""
🐟 **FISH COUNT:** {len(fish_detections)}

⚠️ **Scale Status:** NOT DETECTED

❌ Cannot measure fish length without a visible scale/ruler in the image.

**Tips:**
- Make sure the ruler/scale is clearly visible
- Ensure good lighting on the scale
- Try taking a new photo with the scale more prominent
        """
    
    elif len(fish_detections) == 0:
        # No fish detected
        results_text = f"""
❌ **No fish detected in the image**

**Possible reasons:**
- Image quality is too low
- Fish are too small or unclear
- Lighting is poor
- Model needs more training data

**Try:**
- Taking a clearer photo
- Better lighting
- Moving camera closer to fish
        """
    
    else:
        results_text = "❌ No detections found"
    
    return annotated_image, results_text


# Create Gradio interface
demo = gr.Interface(
    fn=analyze_fish,
    inputs=[
        gr.Image(label="📸 Upload Fish Photo", type="numpy"),
        gr.Number(
            value=10, 
            label="📏 Scale Length (cm)", 
            info="How many centimeters does your ruler/scale show?",
            minimum=1,
            maximum=100
        )
    ],
    outputs=[
        gr.Image(label="🔍 Detection Results", type="numpy"),
        gr.Textbox(
            label="📊 Analysis Results", 
            lines=15,
            show_copy_button=True
        )
    ],
    title="🐟 Fish Counter & Measurement Tool",
    description="""
    **Upload a photo of fish in a tray with a visible ruler/scale.**
    
    The AI will automatically:
    • Count all visible fish
    • Detect the measurement scale
    • Calculate the length of each fish in centimeters
    
    📝 **Instructions:**
    1. Upload your fish photo
    2. Enter the length of your scale/ruler in cm
    3. Click Submit
    4. View results with bounding boxes and measurements
    """,
    article="""
    ### About This Tool
    This fish detection and measurement tool uses YOLOv8 computer vision model trained on custom fish dataset.
    
    **Developed as a school project for automated fish monitoring.**
    """,
    theme="soft",
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 STARTING FISH COUNTER WEB APP")
    print("=" * 60)
    
    if model is None:
        print("\n⚠️ WARNING: Running without model (demo mode)")
        print("The app will start but won't work until you add best.pt\n")
    
    demo.launch(
        share=False,          # Set to True for public link (temporary)
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860      # Port number
    )