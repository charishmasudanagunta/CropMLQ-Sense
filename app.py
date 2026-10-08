import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
from PIL import Image

# ============================================================
# QISKIT
# ============================================================

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QUANTUM_AVAILABLE = True
except Exception:
    QUANTUM_AVAILABLE = False


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="CropMLQ-Sense",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CropMLQ-Sense")
st.subheader("Multi-Objective modal for Crop Disease Testing and Prediction using ML and Quantum Computing")

st.write(
    "CropMLQ-Sense combines crop sensor data, visual leaf analysis, "
    "machine-learning predictions and a quantum-analysis prototype "
    "for early crop disease and stress detection."
)


# ============================================================
# FILES
# ============================================================

DATA_FILE = "AGRIQ_SENSE_200_dataset.xlsx"
DISEASE_MODEL_FILE = "disease_model.pkl"
RISK_MODEL_FILE = "risk_model.pkl"


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

try:
    disease_model = joblib.load(DISEASE_MODEL_FILE)
    risk_model = joblib.load(RISK_MODEL_FILE)

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error("⚠️ Could not load the trained machine-learning models.")

    st.write(
        "Make sure these files are in the same folder as app.py:"
    )

    st.code(
        "disease_model.pkl\n"
        "risk_model.pkl"
    )

    st.write("Technical error:")
    st.write(e)

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    data = pd.read_excel(
        DATA_FILE,
        sheet_name="200 Samples"
    )

except Exception as e:

    st.error(
        f"⚠️ Could not load {DATA_FILE}"
    )

    st.write(e)

    st.stop()


# ============================================================
# DISEASE REMEDIES
# ============================================================

remedies = {

    "Healthy": (
        "No disease-specific treatment is indicated. "
        "Continue normal irrigation, field sanitation and regular monitoring."
    ),

    "Early Blight": (
        "Remove badly affected leaves, avoid prolonged leaf wetness, "
        "improve air circulation and follow locally approved fungicide "
        "recommendations if disease is confirmed."
    ),

    "Late Blight": (
        "Inspect affected plants immediately, remove severely infected "
        "material where appropriate, reduce prolonged leaf wetness and "
        "follow locally approved late-blight management practices."
    ),

    "Brown Spot": (
        "Maintain balanced irrigation and crop nutrition, remove heavily "
        "affected material where appropriate and follow locally approved "
        "Brown Spot control recommendations."
    ),

    "Leaf Blast": (
        "Avoid excessive nitrogen, maintain appropriate field water "
        "management and follow locally approved blast-management practices."
    ),

    "Common Rust": (
        "Remove severely affected material where practical, improve field "
        "air circulation and follow locally approved rust-management "
        "recommendations."
    ),

    "Leaf Blight": (
        "Remove severely affected leaves where appropriate, improve "
        "ventilation, avoid prolonged leaf wetness and follow locally "
        "approved disease-management recommendations."
    )
}


# ============================================================
# FARMER ACTIONS
# ============================================================

farmer_actions = {

    "Healthy": (
        "Continue normal irrigation, field inspection and crop monitoring."
    ),

    "Early Blight": (
        "Inspect lower leaves, remove severely affected leaves and avoid "
        "overhead irrigation where possible."
    ),

    "Late Blight": (
        "Inspect the crop immediately and isolate/remove severely affected "
        "material where appropriate."
    ),

    "Brown Spot": (
        "Inspect leaves for expanding brown spots and maintain balanced "
        "irrigation and crop nutrition."
    ),

    "Leaf Blast": (
        "Inspect leaves and crop canopy, avoid excessive nitrogen and "
        "monitor the field closely."
    ),

    "Common Rust": (
        "Inspect leaves for rust-like pustules and monitor disease spread "
        "across the field."
    ),

    "Leaf Blight": (
        "Inspect affected leaves, improve air circulation and monitor "
        "whether lesions are spreading."
    )
}


# ============================================================
# FIELD SENSOR DATA
# ============================================================

st.header("📡 Field Sensor Data")

col1, col2, col3 = st.columns(3)

with col1:

    crop = st.selectbox(
        "🌾 Crop",
        ["Tomato", "Rice", "Maize"]
    )

with col2:

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=28.0,
        step=0.1
    )

with col3:

    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=0.1
    )


col4, col5 = st.columns(2)

with col4:

    rainfall = st.number_input(
        "🌧️ Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=1.0
    )

with col5:

    soil_moisture = st.number_input(
        "🌱 Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=0.1
    )


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

input_data = pd.DataFrame({

    "Crop": [crop],

    "Temperature (°C)": [temperature],

    "Humidity (%)": [humidity],

    "Rainfall (mm)": [rainfall],

    "Soil Moisture (%)": [soil_moisture]

})


try:

    predicted_disease = disease_model.predict(
        input_data
    )[0]

    predicted_risk = risk_model.predict(
        input_data
    )[0]

except Exception as e:

    st.error("⚠️ Machine-learning prediction failed.")

    st.write(e)

    st.stop()


predicted_disease = str(predicted_disease)
predicted_risk = str(predicted_risk)


# ============================================================
# NORMALIZE RISK LABEL
# ============================================================

risk_lower = predicted_risk.lower()

if "high" in risk_lower:

    risk_display = "HIGH"

elif "moderate" in risk_lower:

    risk_display = "MODERATE"

elif "low" in risk_lower:

    risk_display = "LOW"

else:

    risk_display = predicted_risk.upper()


# ============================================================
# GENERIC LEAF IMAGE ANALYSIS
# ============================================================

st.header("📷 Crop Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a crop/leaf image",
    type=["jpg", "jpeg", "png"]
)


image_stress_score = 0.0
green_percentage = 0.0
yellow_percentage = 0.0
brown_percentage = 0.0
dark_percentage = 0.0

visual_condition = "No image uploaded"


if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        original_image,
        caption="Uploaded Crop Image",
        width=350
    )

    image_rgb = np.array(
        original_image
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    max_size = 700

    height, width = image_bgr.shape[:2]

    if max(height, width) > max_size:

        scale = max_size / max(height, width)

        image_bgr = cv2.resize(
            image_bgr,
            (
                int(width * scale),
                int(height * scale)
            )
        )

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # --------------------------------------------------------
    # EXCESS GREEN
    # --------------------------------------------------------

    b, g, r = cv2.split(
        image_bgr.astype(np.float32)
    )

    excess_green = (
        2 * g - r - b
    )

    # --------------------------------------------------------
    # INITIAL LEAF MASK
    # --------------------------------------------------------

    green_candidate = (
        (excess_green > 10) &
        (S > 35) &
        (V > 35)
    )

    colored_candidate = (
        (S > 45) &
        (V > 35) &
        (V < 245)
    )

    leaf_mask = (
        green_candidate |
        colored_candidate
    )

    # --------------------------------------------------------
    # REMOVE BORDER
    # --------------------------------------------------------

    border = max(
        2,
        int(min(image_bgr.shape[:2]) * 0.02)
    )

    leaf_mask[
        :border, :
    ] = False

    leaf_mask[
        -border:, :
    ] = False

    leaf_mask[
        :, :border
    ] = False

    leaf_mask[
        :, -border:
    ] = False

    # --------------------------------------------------------
    # MORPHOLOGICAL CLEANING
    # --------------------------------------------------------

    mask_uint8 = (
        leaf_mask.astype(np.uint8) * 255
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_OPEN,
        kernel
    )

    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # KEEP LARGE CONNECTED REGION
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )
    )

    if num_labels > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest_index = np.argmax(
            areas
        ) + 1

        largest_area = stats[
            largest_index,
            cv2.CC_STAT_AREA
        ]

        image_area = (
            image_bgr.shape[0] *
            image_bgr.shape[1]
        )

        if largest_area > image_area * 0.02:

            mask_uint8 = np.where(
                labels == largest_index,
                255,
                0
            ).astype(np.uint8)

    leaf_pixels = (
        mask_uint8 > 0
    )

    total_leaf_pixels = np.sum(
        leaf_pixels
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if total_leaf_pixels < 100:

        st.warning(
            "⚠️ The leaf area could not be detected clearly. "
            "Try an image where the leaf is larger and clearer."
        )

        visual_condition = "Image unclear"

    else:

        # ----------------------------------------------------
        # GREEN
        # ----------------------------------------------------

        green_mask = (

            (H >= 35) &
            (H <= 90) &
            (S >= 45) &
            (V >= 45)

        )

        # ----------------------------------------------------
        # YELLOW
        # ----------------------------------------------------

        yellow_mask = (

            (H >= 20) &
            (H < 38) &
            (S >= 45) &
            (V >= 60)

        )

        # ----------------------------------------------------
        # BROWN
        # ----------------------------------------------------

        brown_mask = (

            (H >= 5) &
            (H < 25) &
            (S >= 65) &
            (V >= 30) &
            (V <= 185)

        )

        # ----------------------------------------------------
        # DARK
        # ----------------------------------------------------

        dark_mask = (

            (V < 70) &
            (S > 40)

        )

        # ----------------------------------------------------
        # COUNT PIXELS
        # ----------------------------------------------------

        green_pixels = np.sum(
            green_mask &
            leaf_pixels
        )

        yellow_pixels = np.sum(
            yellow_mask &
            leaf_pixels
        )

        brown_pixels = np.sum(
            brown_mask &
            leaf_pixels
        )

        dark_pixels = np.sum(
            dark_mask &
            leaf_pixels
        )

        # ----------------------------------------------------
        # PERCENTAGES
        # ----------------------------------------------------

        green_percentage = (
            green_pixels /
            total_leaf_pixels
        ) * 100

        yellow_percentage = (
            yellow_pixels /
            total_leaf_pixels
        ) * 100

        brown_percentage = (
            brown_pixels /
            total_leaf_pixels
        ) * 100

        dark_percentage = (
            dark_pixels /
            total_leaf_pixels
        ) * 100

        green_percentage = np.clip(
            green_percentage,
            0,
            100
        )

        yellow_percentage = np.clip(
            yellow_percentage,
            0,
            100
        )

        brown_percentage = np.clip(
            brown_percentage,
            0,
            100
        )

        dark_percentage = np.clip(
            dark_percentage,
            0,
            100
        )

        # ----------------------------------------------------
        # VISUAL STRESS SCORE
        # ----------------------------------------------------

        image_stress_score = (

            yellow_percentage * 0.35

            +

            brown_percentage * 0.50

            +

            dark_percentage * 0.25

        )

        image_stress_score = float(
            np.clip(
                image_stress_score,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # VISUAL CONDITION
        # ----------------------------------------------------

        if image_stress_score < 15:

            visual_condition = (
                "Healthy-looking / Low visual stress"
            )

        elif image_stress_score < 35:

            visual_condition = (
                "Mild visual stress"
            )

        elif image_stress_score < 60:

            visual_condition = (
                "Moderate visual stress"
            )

        else:

            visual_condition = (
                "High visual stress"
            )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        col_img1, col_img2, col_img3, col_img4 = (
            st.columns(4)
        )

        with col_img1:

            st.metric(
                "🟢 Green Area",
                f"{green_percentage:.1f}%"
            )

        with col_img2:

            st.metric(
                "🟡 Yellow Area",
                f"{yellow_percentage:.1f}%"
            )

        with col_img3:

            st.metric(
                "🟤 Brown/Lesion Area",
                f"{brown_percentage:.1f}%"
            )

        with col_img4:

            st.metric(
                "⚫ Dark Area",
                f"{dark_percentage:.1f}%"
            )

        st.metric(
            "📊 Image Stress Score",
            f"{image_stress_score:.1f}/100"
        )

        if image_stress_score < 15:

            st.success(
                "🟢 Visual condition: "
                + visual_condition
            )

        elif image_stress_score < 35:

            st.warning(
                "🟡 Visual condition: "
                + visual_condition
            )

        else:

            st.error(
                "🔴 Visual condition: "
                + visual_condition
            )

        st.info(
            "ℹ️ The image module estimates visual leaf stress "
            "from color patterns. It does not by itself confirm "
            "a specific plant disease."
        )


# ============================================================
# CROP HEALTH ANALYSIS
# ============================================================

st.header("🔍 Crop Health Analysis")

health_col1, health_col2, health_col3 = st.columns(3)

with health_col1:

    st.write("🦠 **Possible Disease**")

    if predicted_disease == "Healthy":

        st.success("Healthy")

    else:

        st.warning(
            predicted_disease
        )


with health_col2:

    st.write("⚠️ **Risk Level**")

    if risk_display == "LOW":

        st.success("LOW")

    elif risk_display == "MODERATE":

        st.warning("MODERATE")

    else:

        st.error("HIGH")


with health_col3:

    if risk_display == "LOW":

        st.success(
            "🟢 LOW RISK"
        )

    elif risk_display == "MODERATE":

        st.warning(
            "🟡 MODERATE RISK"
        )

    else:

        st.error(
            "🔴 HIGH RISK"
        )


# ============================================================
# MULTIMODAL ANALYSIS
# ============================================================

st.header("🔗 Multimodal Analysis")

st.write(
    "The system combines environmental sensor information "
    "with visual crop-image information."
)

col_sensor, col_visual = st.columns(2)

with col_sensor:

    st.subheader("📡 Sensor Analysis")

    st.write(
        f"Temperature: **{temperature:.1f} °C**"
    )

    st.write(
        f"Humidity: **{humidity:.1f}%**"
    )

    st.write(
        f"Rainfall: **{rainfall:.1f} mm**"
    )

    st.write(
        f"Soil Moisture: **{soil_moisture:.1f}%**"
    )

    st.write(
        f"Sensor-based Disease: **{predicted_disease}**"
    )

    st.write(
        f"Sensor-based Risk: **{risk_display}**"
    )


with col_visual:

    st.subheader("📷 Image Analysis")

    if uploaded_file is not None:

        st.write(
            f"Image Stress Score: "
            f"**{image_stress_score:.1f}/100**"
        )

        st.write(
            f"Visual condition: "
            f"**{visual_condition}**"
        )

        st.write(
            f"Green area: "
            f"**{green_percentage:.1f}%**"
        )

        st.write(
            f"Yellow area: "
            f"**{yellow_percentage:.1f}%**"
        )

        st.write(
            f"Brown/lesion-like area: "
            f"**{brown_percentage:.1f}%**"
        )

    else:

        st.info(
            "Upload a crop image for visual analysis."
        )


# ============================================================
# RECOMMENDED REMEDY
# ============================================================

st.header("💊 Recommended Remedy")

if predicted_disease in remedies:

    st.info(
        remedies[predicted_disease]
    )

else:

    st.info(
        "Monitor the crop and inspect symptoms before "
        "taking disease-specific action."
    )


# ============================================================
# FARMER ACTION
# ============================================================

st.header("👨‍🌾 Recommended Farmer Action")

if predicted_disease in farmer_actions:

    st.write(
        farmer_actions[predicted_disease]
    )

else:

    st.write(
        "Continue regular irrigation, field inspection "
        "and crop monitoring."
    )


# ============================================================
# QISKIT QUANTUM ANALYSIS
# ============================================================

st.header("⚛️ Quantum Analysis")

st.write(
    "A 4-qubit Qiskit quantum circuit encodes four normalized "
    "features using RY rotations and connects the qubits "
    "with CNOT gates."
)


# ============================================================
# NORMALIZE FOUR FEATURES
# ============================================================

temperature_norm = np.clip(
    temperature / 50.0,
    0,
    1
)

humidity_norm = np.clip(
    humidity / 100.0,
    0,
    1
)

soil_norm = np.clip(
    soil_moisture / 100.0,
    0,
    1
)

image_norm = np.clip(
    image_stress_score / 100.0,
    0,
    1
)


quantum_features = np.array([

    temperature_norm,
    humidity_norm,
    soil_norm,
    image_norm

])


# Convert normalized values to rotation angles

angles = (
    quantum_features * np.pi
)


# ============================================================
# RUN QISKIT CIRCUIT
# ============================================================

if QUANTUM_AVAILABLE:

    try:

        # ----------------------------------------------------
        # CREATE 4-QUBIT CIRCUIT
        # ----------------------------------------------------

        qc = QuantumCircuit(4, 4)

        # ----------------------------------------------------
        # FEATURE ENCODING
        # ----------------------------------------------------

        for i in range(4):

            qc.ry(
                float(angles[i]),
                i
            )

        # ----------------------------------------------------
        # ENTANGLEMENT
        # ----------------------------------------------------

        qc.cx(0, 1)

        qc.cx(1, 2)

        qc.cx(2, 3)

        # ----------------------------------------------------
        # MEASUREMENTS
        # ----------------------------------------------------

        qc.measure(
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        )

        # ----------------------------------------------------
        # SIMULATE WITH QISKIT AER
        # ----------------------------------------------------

        simulator = AerSimulator()

        shots = 2048

        job = simulator.run(
            qc,
            shots=shots
        )

        result = job.result()

        counts = result.get_counts()

        # ----------------------------------------------------
        # CALCULATE QUBIT ZERO PROBABILITIES
        # ----------------------------------------------------

        zero_probabilities = []

        for qubit in range(4):

            zero_count = 0

            for bitstring, count in counts.items():

                # Qiskit displays classical bits
                # in reverse order.
                bit = bitstring[::-1][qubit]

                if bit == "0":

                    zero_count += count

            probability_zero = (
                zero_count / shots
            )

            zero_probabilities.append(
                probability_zero
            )


        quantum_output = np.array(
            zero_probabilities
        )


        # ----------------------------------------------------
        # CONVERT TO QUANTUM RISK
        # ----------------------------------------------------

        quantum_risk_components = (
            1 - quantum_output
        )

        quantum_score = float(
            np.mean(
                quantum_risk_components
            ) * 100
        )

        quantum_score = float(
            np.clip(
                quantum_score,
                0,
                100
            )
        )


        # ----------------------------------------------------
        # QUANTUM RISK LEVEL
        # ----------------------------------------------------

        if quantum_score < 30:

            quantum_level = "LOW"

        elif quantum_score < 60:

            quantum_level = "MODERATE"

        else:

            quantum_level = "HIGH"


        # ----------------------------------------------------
        # DISPLAY CIRCUIT
        # ----------------------------------------------------

        st.write(
            "### 4-Qubit Qiskit Circuit"
        )

        st.code(
            qc.draw(
                output="text"
            ),
            language="text"
        )


        # ----------------------------------------------------
        # DISPLAY FEATURES
        # ----------------------------------------------------

        q1, q2, q3, q4 = st.columns(4)

        with q1:

            st.metric(
                "Q1",
                f"{quantum_features[0]:.2f}"
            )

            st.caption(
                "Temperature"
            )

        with q2:

            st.metric(
                "Q2",
                f"{quantum_features[1]:.2f}"
            )

            st.caption(
                "Humidity"
            )

        with q3:

            st.metric(
                "Q3",
                f"{quantum_features[2]:.2f}"
            )

            st.caption(
                "Soil Moisture"
            )

        with q4:

            st.metric(
                "Q4",
                f"{quantum_features[3]:.2f}"
            )

            st.caption(
                "Image Stress"
            )


        # ----------------------------------------------------
        # QUANTUM RISK SCORE
        # ----------------------------------------------------

        st.metric(
            "⚛️ Quantum Risk Score",
            f"{quantum_score:.1f}/100"
        )


        if quantum_level == "LOW":

            st.success(
                f"⚛️ Quantum Risk Level: {quantum_level}"
            )

        elif quantum_level == "MODERATE":

            st.warning(
                f"⚛️ Quantum Risk Level: {quantum_level}"
            )

        else:

            st.error(
                f"⚛️ Quantum Risk Level: {quantum_level}"
            )


        # ----------------------------------------------------
        # QISKIT OUTPUT
        # ----------------------------------------------------

        st.write(
            "Quantum measurement probabilities:"
        )

        st.write(
            np.round(
                quantum_output,
                4
            )
        )


        # ----------------------------------------------------
        # MEASUREMENT COUNTS
        # ----------------------------------------------------

        with st.expander("Show Qiskit measurement counts"):

            st.write(counts)


    except Exception as e:

        st.warning(
            "Quantum simulation could not be executed."
        )

        st.write(e)

else:

    st.warning(
        "Qiskit or Qiskit Aer is not available. "
        "The rest of the CropMLQ-Sense dashboard will still work."
    )


# ============================================================
# FINAL INPUT SUMMARY
# ============================================================

st.header("📊 Input Summary")

summary = {

    "Crop": crop,

    "Temperature (°C)": temperature,

    "Humidity (%)": humidity,

    "Rainfall (mm)": rainfall,

    "Soil Moisture (%)": soil_moisture,

    "ML Predicted Disease": predicted_disease,

    "ML Predicted Risk": risk_display,

    "Image Stress Score":
        round(image_stress_score, 1),

    "Green Area (%)":
        round(green_percentage, 1),

    "Yellow Area (%)":
        round(yellow_percentage, 1),

    "Brown/Lesion Area (%)":
        round(brown_percentage, 1),

    "Visual Condition":
        visual_condition

}


st.dataframe(
    pd.DataFrame(
        summary.items(),
        columns=["Parameter", "Value"]
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("🧠 Model Information")

st.write(
    f"The disease and risk models were trained using "
    f"{len(data)} samples from the CropMLQ-Sense dataset."
)

st.write(
    "Sensor features used by the machine-learning models:"
)

st.write(
    "Crop, Temperature, Humidity, Rainfall and Soil Moisture."
)

st.write(
    "The image module provides an additional visual-stress "
    "signal. The quantum section demonstrates a 4-qubit "
    "Qiskit quantum-simulation workflow using RY encoding "
    "and CNOT entanglement."
)

st.caption(
    "Prototype demonstration only. Image color analysis and "
    "machine-learning predictions should not be treated as "
    "a confirmed agricultural diagnosis. Disease-specific "
    "treatment should be based on field inspection and "
    "locally appropriate agricultural guidance."
)