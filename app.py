from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import io
import contextlib
import subprocess
import uuid
import os
import traceback
from PIL import Image
import pytesseract  

# --------------------------
# Step 1: Load and prepare dataset
# --------------------------
df = pd.read_csv("ai_code_debugger_datasets.csv")
df['combined'] = df['language'] + " " + df['code']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['combined'])
y = (df['efficiency_score'] > 65).astype(int)

# --------------------------
# Step 2: Train and save model
# --------------------------
model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# --------------------------
# Step 3: Create Flask app
# --------------------------
app = Flask(__name__)

def extract_code_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def run_code(lang, code_input):
    output = ""
    try:
        if lang == "Python":
            f = io.StringIO()
            try:
                with contextlib.redirect_stdout(f):
                    exec(code_input, {})
            except Exception:
                error_msg = traceback.format_exc()
                output = f.getvalue() + "\n" + error_msg
            else:
                output = f.getvalue()

        elif lang == "JavaScript":
            filename = f"temp_{uuid.uuid4().hex}.js"
            with open(filename, "w") as f:
                f.write(code_input)
            result_sub = subprocess.run(["node", filename], capture_output=True, text=True, timeout=5, shell=True)
            output = result_sub.stdout + result_sub.stderr
            os.remove(filename)

        elif lang == "C++":
            file_id = uuid.uuid4().hex
            cpp_file = f"{file_id}.cpp"
            exe_file = f"{file_id}.exe"
            try:
                with open(cpp_file, "w") as f:
                    f.write(code_input)
                compile_result = subprocess.run(["g++", cpp_file, "-o", exe_file], capture_output=True, text=True)
                if compile_result.returncode == 0:
                    run_result = subprocess.run([exe_file], capture_output=True, text=True, timeout=5)
                    output = run_result.stdout + run_result.stderr
                else:
                    output = compile_result.stderr
            except subprocess.TimeoutExpired:
                output = "Error: C++ execution timed out."
            except Exception as e:
                output = f"Error while running C++ code: {str(e)}"
            finally:
                for ext in [".cpp", ".exe"]:
                    try:
                        os.remove(f"{file_id}{ext}")
                    except:
                        pass

        else:
            output = "Language execution not supported."
    except Exception as e:
        output = f"Error while executing code: {e}"
    return output

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    output = ""
    extracted_code = ""
    step = "upload"  # Default step: show upload & text area

    if request.method == "POST":
        lang = request.form.get('language', '')
        code_file = request.files.get('code_image')
        code_input = request.form.get('code', '')
        action = request.form.get('action', '')

        if action == "Extract Code":  # When user clicks 'Extract Code'
            if code_file and code_file.filename != '':
                image_path = f"temp_{uuid.uuid4().hex}.png"
                code_file.save(image_path)
                extracted_code = extract_code_from_image(image_path)
                os.remove(image_path)
                step = "edit"  # Now show the extracted code in textarea

        elif action == "Check Efficiency":  # When user clicks 'Check Efficiency'
            model = joblib.load("model.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
            combined_input = lang + " " + code_input
            X_input = vectorizer.transform([combined_input])
            prediction = model.predict(X_input)[0]
            confidence = model.predict_proba(X_input)[0][prediction]

            output = run_code(lang, code_input)

            result = {
                "language": lang,
                "prediction": "Efficient" if prediction else "Inefficient",
                "confidence": round(confidence * 100, 2),
                "output": output,
                "extracted_code": code_input
            }
            step = "done"  # Done checking

    return render_template("index.html", result=result, extracted_code=extracted_code, step=step)

if __name__ == "__main__":
    app.run(debug=True)
