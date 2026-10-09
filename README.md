# CropMLQ-Sense

**Multi-Objective modal for Crop Disease Testing and Prediction using ML and Quantum Computing**

## 1. Selected Problem Statement

* **Problem Statement ID:** Add the official ID provided by the organizers.
* **Problem Statement Title:** Add the official selected title.
* **Team Name:** Quantum Codex

## 2. Proposed Solution

CropMLQ-Sense is a multimodal crop-health analysis prototype combining:

* Classical Machine Learning for possible disease and risk prediction.
* OpenCV for visual leaf-stress analysis.
* IBM Qiskit and Qiskit Aer for a complementary quantum risk-analysis experiment.
* Streamlit for an integrated dashboard and farmer-oriented guidance.

## 3. Technical Workflow

1. Enter crop and environmental data.
2. Upload a leaf image.
3. Use ML to predict possible disease and primary risk.
4. Use OpenCV to estimate visual leaf stress.
5. Encode temperature, humidity, soil moisture and image stress into a four-qubit Qiskit circuit.
6. Simulate the circuit using Qiskit Aer with 2,048 shots.
7. Display the separate ML, visual-analysis and quantum results in the dashboard.

## 4. Technologies Used

Python, Streamlit, Pandas, NumPy, Scikit-learn, OpenCV, Joblib, Qiskit, Qiskit Aer and OpenPyXL.

## 5. Dataset

The project includes an Excel prototype dataset with 200 crop/environment records for Tomato, Rice and Maize. It contains disease labels, risk levels and environmental features.

Development data and predictions require further validation using real agricultural field data.

## 6. Quantum Implementation

The quantum module uses a four-qubit circuit with RY feature-encoding gates, CNOT interactions and measurement. Qiskit Aer simulates the circuit locally; the current implementation does not execute on IBM Quantum cloud hardware.

## 7. Prototype Results and Limitations

The current prototype demonstrates the integrated software workflow. OpenCV provides a visual-stress estimate rather than a confirmed disease diagnosis. Further model evaluation, real-world field testing and agricultural expert validation are required.

## 8. Future Scope

* Expand crop-specific image datasets.
* Integrate real-time agricultural sensors.
* Validate the models under field conditions.
* Compare classical and quantum approaches.
* Explore IBM Quantum hardware execution.

## 9. Team Details

Add each team member's name and project role here.

## 10. References

Add the research papers and Qiskit references cited in the project presentation.

## 11. Project Files

* `app.py` — Streamlit application.
* `train_model.py` — model-training script.
* `disease_model.pkl` — trained disease model.
* `risk_model.pkl` — trained risk model.
* `AGRIQ_SENSE_200_dataset.xlsx` — prototype dataset.
* `requirements.txt` — Python dependencies.
* `START_AGRIQ_SENSE.bat` — Windows launcher.
* Add the project presentation/demo file to this repository.
