# Brain Tumor AI — MRI Image Analysis (Academic Prototype)

A responsive Flask web application inspired by the supplied 11-screen design. The UI uses the same **dark navy + bright blue + white + pale blue** palette throughout, with a custom brain-scan illustration on the home page.

## Run on Windows
1. Install Python 3.10 or 3.11.
2. Extract this ZIP and open the extracted `brain_tumor_ai_project` folder in VS Code.
3. Open Terminal and run:
   ```bash
   python -m venv venv
   venv\\Scripts\\activate
   pip install -r requirements.txt
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.
5. Demo login: **admin** / **admin123**.

## Features
- Consistent navy/blue/white medical UI and responsive layout
- Home page with custom brain scan illustration
- Admin login and dashboard
- SQLite database automatically created on first run
- Patient registration, patient list, profile, and scan history
- JPG/JPEG/PNG MRI upload (maximum 12 MB)
- Prediction-result page, printable report, and specialist guidance page

## AI model note
- The web app runs without TensorFlow or a trained model. Without a trained model, upload results are explicitly labelled **DEMO ONLY — no trained model found**. The fallback result is illustrative and is NOT an AI prediction or medical diagnosis.
- For actual model inference, install a compatible TensorFlow version, place a trained Keras model at `models/brain_tumor_model.keras`, and ensure `models/class_names.json` matches the model output class order.
- `train_model.py` is an optional starter script. Put an appropriately licensed MRI dataset under `dataset/Training/<class>/` and `dataset/Testing/<class>/`, install TensorFlow, then run `python train_model.py`. Evaluate the model properly before presenting accuracy claims.

## Important limitation
This is an educational prototype, not a certified medical device. It must not be used to diagnose, treat, or make decisions about real patients. Do not upload identifiable patient data without appropriate consent and safeguards. Sample patient names are fictional demonstration data.
