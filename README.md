# CropMLQ-Sense

**Multi-Objective modal for Crop Disease Testing and Prediction using ML and Quantum Computing**

## 1. Selected Problem Statement

* **Problem Statement ID:** VNQFF-04
* **Problem Statement Title:** Quantum Multimodal Crop Disease Early Warning
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
Sudanagunta Charishma(Team Lead)
Kondragunta Sripujitha
Siriki Tanusri Durga
Narra Mokshitha Chowdary
Posani Lakshmi Teja Sri

## 10. References
1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science, 7, 1419. DOI: https://doi.org/10.3389/fpls.2016.01419

2. Zhou, J., et al. (2021). *Crop disease identification and interpretation method based on multimodal deep learning.* Computers and Electronics in Agriculture, 189, 106408. DOI: https://doi.org/10.1016/j.compag.2021.106408

3. Sarkar, C., et al. (2023). *Leaf disease detection using machine learning and deep learning: Review and challenges.* Applied Soft Computing, 145, 110534. DOI: https://doi.org/10.1016/j.asoc.2023.110534

4. EQID (2025). *Entangled quantum image descriptor: An approach for early plant disease detection.* Crop Protection, 188, 107005. DOI: https://doi.org/10.1016/j.cropro.2024.107005

5. Kumar, S., et al. (2026). *Fusion of quantum computing with smart agriculture: A systematic review of methods, implementation, applications, and challenges.* Information Fusion, 131, 104159. DOI: https://doi.org/10.1016/j.inffus.2026.104159

6. Aleksandrowicz, G., et al. *Qiskit: An Open-source Framework for Quantum Computing.* Zenodo. DOI: https://doi.org/10.5281/zenodo.2562111


## 11. Project Files

* `app.py` — Streamlit application.
* `train_model.py` — model-training script.
* `disease_model.pkl` — trained disease model.
* `risk_model.pkl` — trained risk model.
* `AGRIQ_SENSE_200_dataset.xlsx` — prototype dataset.
* `requirements.txt` — Python dependencies.
* `START_AGRIQ_SENSE.bat` — Windows launcher.
* Add the project presentation/demo file to this repository.
