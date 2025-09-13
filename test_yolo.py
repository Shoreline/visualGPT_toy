#!/usr/bin/env python3
"""
Quick test script for YOLO object detection
"""

from ultralytics import YOLO
import cv2
from rich.console import Console

console = Console()

def test_yolo():
    console.print("[bold blue]Loading YOLO model...[/bold blue]")
    
    # Load YOLO model (will download if first time)
    model = YOLO('yolov8n.pt')  # nano model - smallest and fastest
    
    console.print("[bold green]Model loaded! Running detection on cat.jpg...[/bold green]")
    
    # Run detection on your cat image
    results = model('cat.jpg')
    
    # Print results
    for result in results:
        console.print(f"[bold yellow]Detected objects:[/bold yellow]")
        if result.boxes is not None:
            for box in result.boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                console.print(f"  - {class_name}: {confidence:.2f}")
        else:
            console.print("  No objects detected")
    
    # Save annotated image
    annotated_img = results[0].plot()
    cv2.imwrite('cat_detected.jpg', annotated_img)
    console.print("[bold green]Annotated image saved as 'cat_detected.jpg'[/bold green]")

if __name__ == "__main__":
    test_yolo()
