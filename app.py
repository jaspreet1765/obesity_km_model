import os
import joblib
import traceback
import gradio as gr

# ==========================================================
# Load Trained Model
# ==========================================================
try:
    model = joblib.load("obesity_knn_model.pkl")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None


# ==========================================================
# Prediction Function
# ==========================================================
def predict_obesity(
    Age,
    Gender,
    Height,
    Weight,
    CALC,
    FAVC,
    FCVC,
    NCP,
    SMOKE,
    CH2O,
    family_history_with_overweight,
    FAF,
    TUE,
    MTRANS
):

    # Check empty fields
    values = [
        Age, Gender, Height, Weight, CALC, FAVC, FCVC,
        NCP, SMOKE, CH2O,
        family_history_with_overweight,
        FAF, TUE, MTRANS
    ]

    if any(v is None or str(v).strip() == "" for v in values):
        return "⚠ Please fill all input fields."

    try:
        Age = float(Age)
        Gender = int(Gender)
        Height = float(Height)
        Weight = float(Weight)
        CALC = int(CALC)
        FAVC = int(FAVC)
        FCVC = float(FCVC)
        NCP = float(NCP)
        SMOKE = int(SMOKE)
        CH2O = float(CH2O)
        family_history_with_overweight = int(
            family_history_with_overweight
        )
        FAF = float(FAF)
        TUE = float(TUE)
        MTRANS = int(MTRANS)

    except Exception:
        return "❌ Invalid input. Please enter correct values."

    # Negative values check
    if Age <= 0 or Height <= 0 or Weight <= 0:
        return "❌ Age, Height and Weight must be greater than zero."

    if model is None:
        return "❌ Model not loaded."

    try:

        input_data = [[
            Age,
            Gender,
            Height,
            Weight,
            CALC,
            FAVC,
            FCVC,
            NCP,
            SMOKE,
            CH2O,
            family_history_with_overweight,
            FAF,
            TUE,
            MTRANS
        ]]

        prediction = model.predict(input_data)[0]

        bmi = Weight / (Height * Height)

        result = f"""
🩺 Obesity Prediction

Predicted Class : {str(prediction).replace('_',' ')}

BMI : {bmi:.2f}
"""

        return result

    except Exception:
        return traceback.format_exc()

# ==========================================================
# Gradio Interface
# ==========================================================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
# 🩺 Obesity Level Estimation System

Estimate obesity level using a trained **K-Nearest Neighbors (KNN)** model.
"""
    )

    with gr.Row():

        with gr.Column():
            Age = gr.Number(label="Age")
            Gender = gr.Dropdown(
                choices=[("Female", 0), ("Male", 1)],
                label="Gender"
            )
            Height = gr.Number(label="Height (m)")
            Weight = gr.Number(label="Weight (kg)")
            family_history_with_overweight = gr.Dropdown(
                choices=[("No", 0), ("Yes", 1)],
                label="Family History"
            )

        with gr.Column():
            CALC = gr.Dropdown(
                choices=[
                    ("No", 0),
                    ("Sometimes", 1),
                    ("Frequently", 2),
                    ("Always", 3)
                ],
                label="Alcohol Consumption"
            )

            FAVC = gr.Dropdown(
                choices=[("No", 0), ("Yes", 1)],
                label="High Calorie Food"
            )

            FCVC = gr.Number(label="Vegetable Consumption (1-3)")
            NCP = gr.Number(label="Main Meals Per Day")
            SMOKE = gr.Dropdown(
                choices=[("No", 0), ("Yes", 1)],
                label="Smoking"
            )

        with gr.Column():
            CH2O = gr.Number(label="Water Intake (Litres)")
            FAF = gr.Number(label="Physical Activity")
            TUE = gr.Number(label="Technology Usage")
            MTRANS = gr.Dropdown(
                choices=[
                    ("Automobile", 0),
                    ("Motorbike", 1),
                    ("Bike", 2),
                    ("Public Transport", 3),
                    ("Walking", 4),
                ],
                label="Transportation"
            )

    predict_btn = gr.Button(
        "Predict Obesity Level",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result",
        lines=5
    )

    predict_btn.click(
        fn=predict_obesity,
        inputs=[
            Age,
            Gender,
            Height,
            Weight,
            CALC,
            FAVC,
            FCVC,
            NCP,
            SMOKE,
            CH2O,
            family_history_with_overweight,
            FAF,
            TUE,
            MTRANS
        ],
        outputs=output
    )

    gr.Markdown(
        """
---
### 👩‍💻 Developed By

**Jaspreet Kaur**


"""
    )


# ==========================================================
# Launch App
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
