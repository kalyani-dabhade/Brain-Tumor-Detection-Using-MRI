import os, json, sqlite3, uuid
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'brain_tumor.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'brain_tumor_model.keras')
CLASS_PATH = os.path.join(BASE_DIR, 'models', 'class_names.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-secret-key-for-production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL, age INTEGER, gender TEXT, phone TEXT DEFAULT '',
            notes TEXT DEFAULT '', registered_at TEXT NOT NULL)''')
        con.execute('''CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL,
            image_path TEXT NOT NULL, prediction TEXT NOT NULL, confidence REAL,
            scan_date TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'demo')''')
        count = con.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        if count == 0:
            sample = [
                ('P001','Sneha Patil',28,'Female'),('P002','Rohit Sharma',34,'Male'),
                ('P003','Priya Desai',41,'Female'),('P004','Akash More',22,'Male'),
                ('P005','Neha Jadhav',36,'Female')]
            for pid, name, age, gender in sample:
                con.execute('INSERT INTO patients(patient_id,name,age,gender,registered_at) VALUES(?,?,?,?,?)',
                            (pid,name,age,gender,'2026-10-20'))

init_db()


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapped


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_if_available():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(MODEL_PATH)
    except Exception:
        return None


def predict_image(filepath):
    """Uses the trained Keras model when installed; otherwise returns an explicitly labelled demo result."""
    model = load_model_if_available()
    if model is not None:
        try:
            from tensorflow.keras.utils import load_img, img_to_array
            img = load_img(filepath, target_size=(224, 224), color_mode='rgb')
            arr = img_to_array(img) / 255.0
            pred = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
            if os.path.exists(CLASS_PATH):
                with open(CLASS_PATH, 'r', encoding='utf-8') as f:
                    classes = json.load(f)
            else:
                classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            idx = int(np.argmax(pred))
            label = classes[idx] if idx < len(classes) else f'Class {idx}'
            return label, float(pred[idx]) * 100, 'AI model'
        except Exception:
            pass
    # A deterministic illustrative result, clearly marked as not an AI/medical result.
    # It keeps the academic interface demonstrable before a real model is added.
    try:
        img = Image.open(filepath).convert('L').resize((32, 32))
        arr = np.asarray(img, dtype=np.float32)
        score = float(arr.std())
        labels = ['Demo: No Tumor', 'Demo: Glioma', 'Demo: Meningioma', 'Demo: Pituitary']
        label = labels[int(score) % len(labels)]
        confidence = 50.0 + (score % 20)
    except Exception:
        label, confidence = 'Demo: Review Required', 50.0
    return label, confidence, 'DEMO ONLY — no trained model found'


@app.context_processor
def inject_globals():
    return {'app_name': 'Brain Tumor AI', 'current_year': datetime.now().year}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    with db() as con:
        total_patients = con.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        total_scans = con.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
        recent = con.execute('''SELECT scans.*, patients.name FROM scans LEFT JOIN patients
            ON scans.patient_id=patients.patient_id ORDER BY scans.id DESC LIMIT 6''').fetchall()
        distribution = con.execute('SELECT prediction, COUNT(*) AS n FROM scans GROUP BY prediction').fetchall()
    return render_template('dashboard.html', total_patients=total_patients, total_scans=total_scans,
                           recent=recent, distribution=distribution)

@app.route('/patients')
@login_required
def patients():
    q = request.args.get('q','').strip()
    with db() as con:
        if q:
            rows = con.execute('SELECT * FROM patients WHERE name LIKE ? OR patient_id LIKE ? ORDER BY id DESC', (f'%{q}%',f'%{q}%')).fetchall()
        else:
            rows = con.execute('SELECT * FROM patients ORDER BY id DESC').fetchall()
    return render_template('patients.html', patients=rows, q=q)

@app.route('/patients/add', methods=['GET','POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        age = request.form.get('age','').strip()
        gender = request.form.get('gender','').strip()
        if not name or not age or not gender:
            flash('Please complete all required fields.', 'error')
        else:
            with db() as con:
                next_num = con.execute('SELECT COUNT(*) FROM patients').fetchone()[0] + 1
                pid = f'P{next_num:03d}'
                while con.execute('SELECT 1 FROM patients WHERE patient_id=?',(pid,)).fetchone():
                    next_num += 1; pid = f'P{next_num:03d}'
                con.execute('INSERT INTO patients(patient_id,name,age,gender,phone,notes,registered_at) VALUES(?,?,?,?,?,?,?)',
                            (pid,name,int(age),gender,request.form.get('phone','').strip(),request.form.get('notes','').strip(),datetime.now().strftime('%Y-%m-%d')))
            flash(f'Patient {pid} registered successfully.', 'success')
            return redirect(url_for('patients'))
    return render_template('add_patient.html')

@app.route('/patients/<patient_id>')
@login_required
def patient_detail(patient_id):
    with db() as con:
        patient = con.execute('SELECT * FROM patients WHERE patient_id=?',(patient_id,)).fetchone()
        scans = con.execute('SELECT * FROM scans WHERE patient_id=? ORDER BY id DESC',(patient_id,)).fetchall()
    if not patient:
        flash('Patient not found.', 'error'); return redirect(url_for('patients'))
    return render_template('patient_detail.html', patient=patient, scans=scans)

@app.route('/mri-analysis', methods=['GET','POST'])
@login_required
def mri_analysis():
    with db() as con:
        patient_rows = con.execute('SELECT patient_id,name FROM patients ORDER BY name').fetchall()
    if request.method == 'POST':
        patient_id = request.form.get('patient_id','').strip()
        file = request.files.get('mri_image')
        if not patient_id or not file or not file.filename:
            flash('Select a patient and an MRI image.', 'error')
            return redirect(url_for('mri_analysis'))
        if not allowed_file(file.filename):
            flash('Upload a PNG, JPG, or JPEG image.', 'error')
            return redirect(url_for('mri_analysis'))
        with db() as con:
            patient = con.execute('SELECT * FROM patients WHERE patient_id=?',(patient_id,)).fetchone()
        if not patient:
            flash('Selected patient does not exist.', 'error'); return redirect(url_for('mri_analysis'))
        filename = f'{uuid.uuid4().hex}_{secure_filename(file.filename)}'
        disk_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(disk_path)
        label, confidence, mode = predict_image(disk_path)
        rel_path = 'uploads/' + filename
        with db() as con:
            cur = con.execute('INSERT INTO scans(patient_id,image_path,prediction,confidence,scan_date,mode) VALUES(?,?,?,?,?,?)',
                (patient_id, rel_path, label, confidence, datetime.now().strftime('%Y-%m-%d %H:%M'), mode))
            scan_id = cur.lastrowid
        return redirect(url_for('result', scan_id=scan_id))
    return render_template('mri_analysis.html', patients=patient_rows, model_available=os.path.exists(MODEL_PATH))

@app.route('/result/<int:scan_id>')
@login_required
def result(scan_id):
    with db() as con:
        scan = con.execute('''SELECT scans.*, patients.name, patients.age, patients.gender FROM scans
            LEFT JOIN patients ON scans.patient_id=patients.patient_id WHERE scans.id=?''',(scan_id,)).fetchone()
    if not scan:
        flash('Result not found.', 'error'); return redirect(url_for('history'))
    return render_template('result.html', scan=scan)

@app.route('/history')
@login_required
def history():
    q = request.args.get('q','').strip()
    with db() as con:
        sql = '''SELECT scans.*, patients.name FROM scans LEFT JOIN patients ON scans.patient_id=patients.patient_id'''
        if q:
            rows = con.execute(sql+' WHERE scans.patient_id LIKE ? OR patients.name LIKE ? ORDER BY scans.id DESC', (f'%{q}%',f'%{q}%')).fetchall()
        else:
            rows = con.execute(sql+' ORDER BY scans.id DESC').fetchall()
    return render_template('history.html', scans=rows, q=q)

@app.route('/report/<int:scan_id>')
@login_required
def report(scan_id):
    with db() as con:
        scan = con.execute('''SELECT scans.*, patients.name, patients.age, patients.gender FROM scans
            LEFT JOIN patients ON scans.patient_id=patients.patient_id WHERE scans.id=?''',(scan_id,)).fetchone()
    if not scan:
        flash('Report not found.', 'error'); return redirect(url_for('history'))
    return render_template('report.html', scan=scan)

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(413)
def too_large(_):
    flash('Image is too large. Maximum upload size is 12 MB.', 'error')
    return redirect(url_for('mri_analysis'))

if __name__ == '__main__':
    app.run(debug=True)
