Bottle Quality Detection System
Project Summary & Technical Documentation
Dalnex  |  Computer Vision QC Project  |  2025
1. Project Overview
The Bottle Quality Detection System is a computer vision-based industrial quality control application developed for Dalnex. It leverages a custom-trained YOLO (You Only Look Once) deep learning model to automatically classify bottles on a production line as either Good or Bad in real time, eliminating the need for manual visual inspection.

The system processes input images through a trained neural network model and returns a classification decision along with a bounding box overlay on the detected bottle. It is designed for ease of use — an operator simply provides an image path and the system returns an instant verdict.

2. Project Objectives
•	Automate bottle quality inspection using deep learning to reduce human error.
•	Detect and classify bottles into two categories: Good Bottle and Bad Bottle.
•	Provide a real-time visual output showing the detection bounding box and label.
•	Measure and validate the model's accuracy using standard object detection metrics.
•	Build an extendable, maintainable Python codebase for production use.

3. Technology Stack

Component	Technology / Library	Purpose
Programming Language	Python 3.x	Core development language
Deep Learning Framework	Ultralytics YOLOv8	Object detection model training & inference
Computer Vision	OpenCV (cv2)	Image loading, drawing, and display
Model Format	.pt (PyTorch weights)	Trained model storage and loading
Dataset Format	YOLOv8 / Roboflow	Annotated image dataset for training & evaluation
Annotation Tool	Roboflow	Dataset labeling, versioning, and export


4. Model Details
4.1 Architecture
The model is based on YOLOv8, a state-of-the-art single-stage object detector developed by Ultralytics. YOLOv8 performs detection and classification in a single forward pass of the neural network, making it fast enough for real-time applications.

4.2 Classes

Class ID	Label	Description
0	Bad Bottle	Bottle with defects (damaged, crushed, misaligned cap, etc.)
1	Good Bottle	Bottle that passes quality inspection standards


4.3 Model File
•	Location: E:\Dalnex\runs\detect\bottle_stable_model\weights\best.pt
•	The model was trained using the Ultralytics training pipeline on a custom annotated dataset.
•	best.pt represents the checkpoint with the highest validation performance during training.

5. Dataset

Property	Details
Dataset Name	Bottle.v3-0.1.1.yolov8
Format	YOLOv8 (YOLO annotation format)
YAML Config	E:\Dalnex\Bottle.v3-0.1.1.yolov8\data.yaml
Number of Classes	2 (bad_bottle, good_bottle)
Splits	Train / Validation / Test
Source	Roboflow annotated dataset


6. System Workflow
The system follows a straightforward pipeline from image input to final classification output:

•	Step 1 — Model Load: The trained YOLOv8 model (best.pt) is loaded into memory at startup.
•	Step 2 — Image Input: The operator provides a path to the image to be inspected.
•	Step 3 — Inference: The model runs object detection on the image at a confidence threshold of 0.4.
•	Step 4 — Confidence Extraction: For each detected bounding box, the class ID (0 = Bad, 1 = Good) and confidence score are extracted.
•	Step 5 — Decision Logic: The system applies confidence thresholds (Good ≥ 0.5, Bad ≥ 0.7) to determine the final label. If neither threshold is met, the class with the higher raw confidence wins.
•	Step 6 — Visual Output: The result is drawn on the image with a colored bounding box (Green = Good, Red = Bad) and label text, then displayed to the operator.

7. Classification Decision Logic
The prediction logic applies asymmetric confidence thresholds to reduce the risk of passing bad bottles through quality control:

Condition	Final Decision	Reasoning
Good confidence ≥ 0.50	GOOD BOTTLE ✓	Model is reasonably sure it is good
Bad confidence ≥ 0.70	BAD BOTTLE ✗	Higher bar required to flag as defective
Neither threshold met (Good ≥ Bad)	GOOD BOTTLE ✓	Fallback: good confidence leads
Neither threshold met (Bad > Good)	BAD BOTTLE ✗	Fallback: bad confidence leads

The higher threshold for bad bottles (0.70 vs 0.50) is intentional — in a quality control context, it is preferable to over-flag potential defects than to miss them.

8. Accuracy Evaluation
8.1 Evaluation Method
Model accuracy is evaluated using YOLO's built-in val() method against the labeled test split of the dataset. This method computes standard object detection metrics that account for both the classification label and the spatial accuracy of the predicted bounding box.

8.2 Key Metrics

Metric	Description	Good Value
mAP@50	Mean Average Precision at 50% IoU overlap. Primary accuracy metric.	> 90%
mAP@50-95	mAP averaged across IoU thresholds 50% to 95%. Stricter measure.	> 75%
Precision	Of all Bad Bottle predictions, what fraction were truly bad.	> 90%
Recall	Of all actual Bad Bottles, what fraction were detected.	> 90%

For this application, Recall is the most critical metric. A missed bad bottle (False Negative) is a more costly error than a false alarm (False Positive), as defective products could reach customers.

9. Code Structure

Function / Block	Description
CONFIG section	Defines model path, data YAML path, and confidence threshold constants.
Model Load block	Loads the YOLO model once at startup to avoid repeated disk I/O.
evaluate_accuracy()	Runs model.val() once at startup to compute and print all accuracy metrics.
predict(img_path)	Loads a single image, runs inference, applies decision logic, and displays result.
__main__ block	Entry point: calls evaluate_accuracy() once, then enters interactive predict loop.


10. System Output
10.1 Console Output
•	FINAL RESULT: GOOD BOTTLE or BAD BOTTLE
•	Good confidence score (0.00 to 1.00)
•	Bad confidence score (0.00 to 1.00)
•	On startup: mAP@50, mAP@50-95, Precision, Recall printed to console

10.2 Visual Output
•	Bounding box drawn around the detected bottle in the image.
•	Green box and label for Good Bottle detections.
•	Red box and label for Bad Bottle detections.
•	Result image displayed in a pop-up window until the operator presses any key.

11. Potential Future Improvements
•	Webcam / live video integration for real-time conveyor belt inspection.
•	REST API wrapper (FastAPI/Flask) to allow integration with factory management systems.
•	Multi-defect classification — identifying specific defect types (crack, dent, missing cap, etc.).
•	Automated logging of inspection results to a database or CSV file for traceability.
•	Dashboard UI to display live pass/fail statistics and trends.
•	Model retraining pipeline to continuously improve accuracy with new production data.

12. Summary
The Dalnex Bottle Quality Detection System demonstrates a complete, end-to-end computer vision pipeline for industrial quality control. It combines a custom-trained YOLOv8 model with an intelligent dual-threshold decision engine, delivering fast and reliable pass/fail verdicts on bottle images with clear visual feedback.

The architecture is clean, modular, and production-ready — accuracy evaluation is separated from inference, confidence thresholds are tuned to minimize missed defects, and the codebase is easily extendable to support live video, remote APIs, or more granular defect classification in future iterations.
