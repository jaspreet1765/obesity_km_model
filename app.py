import os
import joblib
import traceback
import gradio as gr

# ==========================================================
# Load Trained Model
# ==========================================================

try:
    deployed_knn = joblib.load("obesity_knn_model.pkl")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(e)
    deployed_knn = None


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
    MTRANS,
):

    values = [
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
        MTRANS,
    ]

    # Empty Input Check
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill all the fields."

    # Type Conversion
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

    except Exception as e:
        return f"❌ Invalid Input\n\n{str(e)}"

    # Negative Value Check
    if (
        Age < 0
        or Height < 0
        or Weight < 0
        or FCVC < 0
        or NCP < 0
        or CH2O < 0
        or FAF < 0
        or TUE < 0
    ):
        return "❌ Negative values are not allowed."

    if deployed_knn is None:
        return "❌ Model failed to load."

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

        prediction = deployed_knn.predict(input_data)

        category_map = {
            0: "Insufficient Weight (Underweight)",
            1: "Normal Weight",
            2: "Overweight Level I",
            3: "Overweight Level II",
            4: "Obesity Type I",
            5: "Obesity Type II",
            6: "Obesity Type III"
        }

        result = category_map.get(
            prediction[0],
            f"Unknown Category ({prediction[0]})"
        )

        return f"""
🩺 Obesity Assessment Result

Predicted Category:

{result}

⚠️ This prediction is generated using a Machine Learning (KNN) model and should not replace professional medical advice.
"""

    except Exception:

        error = traceback.format_exc()
        print(error)

        return f"""
❌ Prediction Failed

{error}
"""


# ==========================================================
# Gradio UI
# ==========================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    )
) as app:

    gr.Markdown(
        """
# 🩺 Obesity Level Estimation System

Estimate obesity level using a trained **K-Nearest Neighbors (KNN)** Machine Learning model.

Fill all the information below and click **Evaluate Obesity Level**.
"""
    )

    with gr.Row():

        # Column 1
        with gr.Column():

            gr.Markdown("## 👤 Personal Details")

            Age = gr.Number(label="Age (Years)")

            Gender = gr.Dropdown(
                choices=[
                    ("Female", 0),
                    ("Male", 1)
                ],
                label="Gender"
            )

            Height = gr.Number(
                label="Height (Meters)"
            )

            Weight = gr.Number(
                label="Weight (Kilograms)"
            )

            family_history_with_overweight = gr.Dropdown(
                choices=[
                    ("No", 0),
                    ("Yes", 1)
                ],
                label="Family History of Overweight"
            )

        # Column 2
        with gr.Column():

            gr.Markdown("## 🥗 Dietary Habits")

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
                choices=[
                    ("No", 0),
                    ("Yes", 1)
                ],
                label="High Calorie Food"
            )

            FCVC = gr.Number(
                label="Vegetable Consumption (1-3)"
            )

            NCP = gr.Number(
                label="Main Meals Per Day"
            )

            CH2O = gr.Number(
                label="Daily Water Intake (Litres)"
            )

        # Column 3
        with gr.Column():

            gr.Markdown("## 🏃 Lifestyle")

            SMOKE = gr.Dropdown(
                choices=[
                    ("No", 0),
                    ("Yes", 1)
                ],
                label="Smoker"
            )

            FAF = gr.Number(
                label="Physical Activity Frequency (0-3)"
            )

            TUE = gr.Number(
                label="Technology Usage Time (0-2)"
            )

            MTRANS = gr.Dropdown(
                choices=[
                    ("Automobile", 0),
                    ("Motorbike", 1),
                    ("Bike", 2),
                    ("Public Transportation", 3),
                    ("Walking", 4)
                ],
                label="Transportation"
            )

    gr.Markdown("---")

    with gr.Row():

        submit_btn = gr.Button(
            "🔍 Evaluate Obesity Level",
            variant="primary"
        )

        clear_btn = gr.ClearButton()

    result_box = gr.Textbox(
        label="Prediction Result",
        lines=8,
        interactive=False
    )

    gr.Markdown(
        """
---

### 👨‍💻 About

**Obesity Level Estimation System**

Developed using **Machine Learning (K-Nearest Neighbors)** and **Gradio**.

⚠️ This application is for educational purposes only and should not replace medical consultation.
"""
    )

    input_components = [
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
    ]

    submit_btn.click(
        fn=predict_obesity,
        inputs=input_components,
        outputs=result_box
    )

    clear_btn.add(
        input_components + [result_box]
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
