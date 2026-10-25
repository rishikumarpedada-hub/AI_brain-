import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
# Vercel's deployed source directory is read-only. Runtime writes must go to /tmp.
IS_VERCEL = bool(os.environ.get("VERCEL"))
RUNTIME_DIR = Path("/tmp/ai-brain") if IS_VERCEL else BASE_DIR
DB_PATH = RUNTIME_DIR / "instance" / "brain_app.db"
MODEL_PATH = BASE_DIR / "dbim_model.joblib"
KEY_PATH = RUNTIME_DIR / "instance" / "data.key"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
GENERATED_DIR = RUNTIME_DIR / "static" / "generated"

# Do not rely on a personal Windows/Cursor path in production.
BACKGROUND_ASSET_DIR = BASE_DIR / "static" / "images"
BACKGROUND_ASSET_FILENAME = "background.png"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", secrets.token_hex(32))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _xor_bytes(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


class SecureCipher:
    def __init__(self, master_key: bytes):
        self.enc_key = hmac.new(master_key, b"enc", hashlib.sha256).digest()
        self.mac_key = hmac.new(master_key, b"mac", hashlib.sha256).digest()

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hmac.new(self.enc_key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    def encrypt(self, plaintext: str) -> str:
        data = plaintext.encode("utf-8")
        nonce = secrets.token_bytes(16)
        ct = _xor_bytes(data, self._keystream(nonce, len(data)))
        tag = hmac.new(self.mac_key, nonce + ct, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(nonce + ct + tag).decode("utf-8")

    def decrypt(self, token: str) -> str:
        raw = base64.urlsafe_b64decode(token.encode("utf-8"))
        nonce, body = raw[:16], raw[16:]
        ct, tag = body[:-32], body[-32:]
        expected = hmac.new(self.mac_key, nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("Data integrity check failed")
        return _xor_bytes(ct, self._keystream(nonce, len(ct))).decode("utf-8")


def init_cipher():
    env_key = os.environ.get("DATA_ENCRYPTION_KEY")
    if env_key:
        master_key = hashlib.sha256(env_key.encode("utf-8")).digest()
        return SecureCipher(master_key)

    if KEY_PATH.exists():
        master_key = KEY_PATH.read_bytes().strip()
        return SecureCipher(master_key)

    master_key = secrets.token_bytes(32)
    KEY_PATH.write_bytes(master_key)
    return SecureCipher(master_key)


cipher = init_cipher()
loaded_model = joblib.load(MODEL_PATH)
if isinstance(loaded_model, dict) and "model" in loaded_model:
    model = loaded_model["model"]
    label_to_name = loaded_model.get("label_to_name", {0: "non-stroke", 1: "stroke"})
else:
    model = loaded_model
    label_to_name = {0: "non-stroke", 1: "stroke"}


def hash_password(password: str, salt: bytes | None = None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return base64.b64encode(salt).decode("utf-8"), base64.b64encode(digest).decode("utf-8")


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64.encode("utf-8"))
    expected = base64.b64decode(hash_b64.encode("utf-8"))
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(actual, expected)


def encrypt_text(value: str) -> str:
    return cipher.encrypt(value)


def decrypt_text(value: str) -> str:
    return cipher.decrypt(value)


def normalize_prediction_label(label: str) -> str:
    if not label:
        return label
    if label.strip().lower().startswith("stroke"):
        return "Stroke"
    return label


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            patient_name_enc TEXT NOT NULL,
            patient_age INTEGER NOT NULL,
            patient_gender TEXT NOT NULL,
            notes_enc TEXT,
            image_filename TEXT NOT NULL,
            overlay_filename TEXT NOT NULL,
            highlight_filename TEXT,
            prediction_label TEXT NOT NULL,
            prediction_confidence REAL,
            lesion_ratio REAL,
            lesion_pixels INTEGER,
            total_pixels INTEGER,
            region_count INTEGER,
            largest_region_ratio REAL,
            feature_mean REAL,
            feature_std REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
        """
    )

    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(patients)").fetchall()
    }
    needed_columns = {
        "prediction_confidence": "REAL",
        "lesion_ratio": "REAL",
        "lesion_pixels": "INTEGER",
        "total_pixels": "INTEGER",
        "highlight_filename": "TEXT",
        "region_count": "INTEGER",
        "largest_region_ratio": "REAL",
        "feature_mean": "REAL",
        "feature_std": "REAL",
    }
    for col, col_type in needed_columns.items():
        if col not in existing_columns:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {col} {col_type}")

    doctor = conn.execute("SELECT id FROM doctors LIMIT 1").fetchone()
    if not doctor:
        username = os.environ.get("DOCTOR_USERNAME", "doctor")
        password = os.environ.get("DOCTOR_PASSWORD", "changeme123")
        salt, pwd_hash = hash_password(password)
        conn.execute(
            "INSERT INTO doctors (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, salt, pwd_hash, datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_mmi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return filtered


def forward_model(permittivity_map):
    return np.fft.fft2(permittivity_map)


def dbim_segmentation(mmi_data, iterations=5, step_size=0.01):
    permittivity_map = np.ones_like(mmi_data, dtype=np.float32)
    measured_fields = np.fft.fft2(mmi_data)

    for _ in range(iterations):
        simulated_fields = forward_model(permittivity_map)
        error = measured_fields - simulated_fields
        permittivity_map += step_size * np.real(np.fft.ifft2(error))

    min_v = permittivity_map.min()
    max_v = permittivity_map.max()
    denom = max(max_v - min_v, 1e-8)
    permittivity_map = (permittivity_map - min_v) / denom
    mask = (permittivity_map > 0.5).astype(np.uint8)
    return mask


def extract_features(image):
    hist = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    mean_val = np.mean(image)
    std_val = np.std(image)
    return np.concatenate([hist, [mean_val, std_val]])


def extract_lesion_regions(preprocessed_img, dbim_mask):
    # 1) Brain-only mask (largest connected soft-tissue region), then shrink inward to remove skull ring.
    _, brain_bin = cv2.threshold(preprocessed_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    brain_bin = cv2.morphologyEx(brain_bin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(brain_bin, connectivity=8)
    brain_mask = np.zeros_like(brain_bin, dtype=np.uint8)
    if nb > 1:
        largest_idx = 1 + int(np.argmax(sb[1:, cv2.CC_STAT_AREA]))
        brain_mask[lb == largest_idx] = 1
    else:
        brain_mask = brain_bin.copy()
    inner_brain = cv2.erode(brain_mask, np.ones((11, 11), np.uint8), iterations=2)

    # 2) Bright-lesion candidates inside inner brain.
    smooth = cv2.GaussianBlur(preprocessed_img, (5, 5), 0)
    roi_vals = smooth[inner_brain > 0] if np.any(inner_brain > 0) else smooth.reshape(-1)
    p88 = float(np.percentile(roi_vals, 88))
    adaptive_thr = float(np.mean(roi_vals) + 0.55 * np.std(roi_vals))
    bright_thr = max(p88, adaptive_thr)
    bright_mask = ((smooth >= bright_thr) & (inner_brain > 0)).astype(np.uint8)

    # 3) Use DBIM only as a soft prior to strengthen overlap, not to flood-fill.
    dbim_in = cv2.bitwise_and(dbim_mask.astype(np.uint8), inner_brain)
    combined = cv2.bitwise_or(bright_mask, cv2.bitwise_and(bright_mask, dbim_in))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    # 4) Keep one dominant internal component (largest bright lesion-like region).
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    filtered = np.zeros_like(combined, dtype=np.uint8)
    total_pixels = int(combined.size)
    min_area = max(45, int(total_pixels * 0.0025))
    max_area = int(total_pixels * 0.30)
    h, w = combined.shape
    border_margin = max(6, int(min(h, w) * 0.04))

    best_idx = None
    best_score = -1.0
    for idx in range(1, nl):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        touches_border = x <= border_margin or y <= border_margin or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin)
        if touches_border:
            continue
        if area < min_area or area > max_area:
            continue

        comp_mask = labels == idx
        mean_intensity = float(np.mean(smooth[comp_mask])) if np.any(comp_mask) else 0.0
        fill_ratio = area / max(bw * bh, 1)
        norm_area = area / max(total_pixels, 1)
        score = norm_area * 0.60 + (mean_intensity / 255.0) * 0.30 + fill_ratio * 0.10
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        filtered[labels == best_idx] = 1
        region_count = 1
        largest_area = int(stats[best_idx, cv2.CC_STAT_AREA])
    else:
        region_count = 0
        largest_area = 0

    largest_region_ratio = float(largest_area / max(total_pixels, 1))
    return filtered, region_count, largest_region_ratio


def predict_and_save_overlay(img_path: Path):
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError("Invalid image file")

    pre = preprocess_mmi(image)
    seg = dbim_segmentation(pre)
    seg, region_count, largest_region_ratio = extract_lesion_regions(pre, seg)
    features = extract_features(seg).reshape(1, -1)
    prediction = model.predict(features)[0]
    raw_label = label_to_name.get(int(prediction), str(prediction))
    pretty_map = {
        "non-stroke": "Non-Stroke",
        "nonstroke": "Non-Stroke",
        "stroke": "Stroke",
        "ischemic": "Ischemic Stroke",
        "hemorrhagic": "Hemorrhagic Stroke",
    }
    label = pretty_map.get(raw_label, raw_label.replace("_", " ").title())
    confidence = None
    stroke_probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(np.max(proba))
        if hasattr(model, "classes_"):
            for idx, class_id in enumerate(model.classes_):
                class_name = str(label_to_name.get(int(class_id), class_id)).lower().replace("_", "-")
                if "stroke" in class_name and "non" not in class_name:
                    stroke_probability = float(proba[idx])
                    break

    lesion_ratio = float(np.mean(seg))
    lesion_pixels = int(np.sum(seg))
    total_pixels = int(seg.size)
    feature_mean = float(np.mean(seg))
    feature_std = float(np.std(seg))

    # Rule-assisted safeguard for obvious lesion cases that are predicted as Non-Stroke.
    is_non_stroke = label.lower().strip() == "non-stroke"
    strong_lesion_evidence = (
        region_count >= 1
        and lesion_ratio >= 0.03
        and largest_region_ratio >= 0.01
    )
    moderate_lesion_evidence = (
        region_count >= 1
        and lesion_ratio >= 0.02
        and largest_region_ratio >= 0.007
        and stroke_probability is not None
        and stroke_probability >= 0.22
    )
    if is_non_stroke and (strong_lesion_evidence or moderate_lesion_evidence):
        label = "Stroke"

    seg_u8 = (seg * 255).astype(np.uint8)
    overlay = image.copy()
    red_layer = np.zeros_like(image, dtype=np.uint8)
    red_layer[:, :, 2] = 255
    overlay = np.where(seg_u8[:, :, None] > 0, cv2.addWeighted(overlay, 0.55, red_layer, 0.45, 0), overlay)
    highlighted = image.copy()
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_idx = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        region_mask = np.zeros(seg_u8.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        highlighted = np.where(
            region_mask[:, :, None] > 0,
            cv2.addWeighted(highlighted, 0.5, red_layer, 0.5, 0),
            highlighted,
        )
        cv2.drawContours(highlighted, [contour], -1, (0, 0, 255), 2)
        valid_idx += 1

    overlay_name = f"overlay_{img_path.stem}_{secrets.token_hex(5)}.png"
    overlay_path = GENERATED_DIR / overlay_name
    cv2.imwrite(str(overlay_path), overlay)
    highlight_name = f"highlight_{img_path.stem}_{secrets.token_hex(5)}.png"
    highlight_path = GENERATED_DIR / highlight_name
    cv2.imwrite(str(highlight_path), highlighted)

    return (
        label,
        overlay_name,
        highlight_name,
        {
            "confidence": confidence,
            "lesion_ratio": lesion_ratio,
            "lesion_pixels": lesion_pixels,
            "total_pixels": total_pixels,
            "region_count": region_count,
            "largest_region_ratio": largest_region_ratio,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
    )


def login_required(fn):
    def wrapped(*args, **kwargs):
        if not session.get("doctor_id"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    wrapped.__name__ = fn.__name__
    return wrapped


@app.route("/", methods=["GET", "POST"])
def login():
    if session.get("doctor_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        doctor = conn.execute("SELECT * FROM doctors WHERE username = ?", (username,)).fetchone()
        conn.close()

        if doctor and verify_password(password, doctor["salt"], doctor["password_hash"]):
            session["doctor_id"] = doctor["id"]
            session["doctor_username"] = doctor["username"]
            return redirect(url_for("dashboard"))

        flash("Invalid credentials", "error")

    return render_template("login.html")


@app.route("/newdata", methods=["GET", "POST"])
def register_doctor():
    if session.get("doctor_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return redirect(url_for("register_doctor"))

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return redirect(url_for("register_doctor"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register_doctor"))

        conn = get_db_connection()
        exists = conn.execute("SELECT id FROM doctors WHERE username = ?", (username,)).fetchone()
        if exists:
            conn.close()
            flash("Username already exists.", "error")
            return redirect(url_for("register_doctor"))

        salt, pwd_hash = hash_password(password)
        conn.execute(
            "INSERT INTO doctors (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, salt, pwd_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        flash("Doctor account created. Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        name = request.form.get("patient_name", "").strip()
        age = request.form.get("patient_age", "").strip()
        gender = request.form.get("patient_gender", "").strip()
        notes = request.form.get("notes", "").strip()
        file = request.files.get("scan_image")

        if not name or not age or not gender or not file or not allowed_file(file.filename):
            flash("Provide valid patient data and a scan image (png/jpg/jpeg/bmp).", "error")
            return redirect(url_for("dashboard"))

        filename = secure_filename(file.filename)
        stored_name = f"scan_{secrets.token_hex(6)}_{filename}"
        upload_path = UPLOAD_DIR / stored_name
        file.save(upload_path)

        try:
            prediction_label, overlay_filename, highlight_filename, explain = predict_and_save_overlay(upload_path)
        except ValueError:
            upload_path.unlink(missing_ok=True)
            flash("Image processing failed. Upload a valid scan image.", "error")
            return redirect(url_for("dashboard"))

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO patients (
                doctor_id, patient_name_enc, patient_age, patient_gender, notes_enc,
                image_filename, overlay_filename, highlight_filename, prediction_label,
                prediction_confidence, lesion_ratio, lesion_pixels, total_pixels,
                region_count, largest_region_ratio, feature_mean, feature_std, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["doctor_id"],
                encrypt_text(name),
                int(age),
                gender,
                encrypt_text(notes) if notes else None,
                stored_name,
                overlay_filename,
                highlight_filename,
                prediction_label,
                explain["confidence"],
                explain["lesion_ratio"],
                explain["lesion_pixels"],
                explain["total_pixels"],
                explain["region_count"],
                explain["largest_region_ratio"],
                explain["feature_mean"],
                explain["feature_std"],
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        flash("Patient saved and prediction completed.", "success")
        return redirect(url_for("patients"))

    return render_template("dashboard.html", doctor=session.get("doctor_username"))


@app.route("/patients")
@login_required
def patients():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, patient_name_enc, patient_age, patient_gender, prediction_label, prediction_confidence, created_at FROM patients WHERE doctor_id = ? ORDER BY id DESC",
        (session["doctor_id"],),
    ).fetchall()
    conn.close()

    records = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "name": decrypt_text(row["patient_name_enc"]),
                "age": row["patient_age"],
                "gender": row["patient_gender"],
                "prediction": normalize_prediction_label(row["prediction_label"]),
                "confidence": row["prediction_confidence"],
                "created_at": row["created_at"],
            }
        )

    return render_template("patients.html", records=records)


@app.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT * FROM patients
        WHERE id = ? AND doctor_id = ?
        """,
        (patient_id, session["doctor_id"]),
    ).fetchone()
    conn.close()

    if not row:
        flash("Patient not found.", "error")
        return redirect(url_for("patients"))

    data = {
        "id": row["id"],
        "name": decrypt_text(row["patient_name_enc"]),
        "age": row["patient_age"],
        "gender": row["patient_gender"],
        "notes": decrypt_text(row["notes_enc"]) if row["notes_enc"] else "",
        "prediction": normalize_prediction_label(row["prediction_label"]),
        "confidence": row["prediction_confidence"],
        "lesion_ratio": row["lesion_ratio"],
        "lesion_pixels": row["lesion_pixels"],
        "total_pixels": row["total_pixels"],
        "feature_mean": row["feature_mean"],
        "feature_std": row["feature_std"],
        "region_count": row["region_count"],
        "largest_region_ratio": row["largest_region_ratio"],
        "created_at": row["created_at"],
        "overlay_filename": row["overlay_filename"],
        "highlight_filename": row["highlight_filename"],
    }
    return render_template("patient_detail.html", patient=data)


@app.route("/generated/<path:filename>")
@login_required
def generated(filename):
    return send_from_directory(GENERATED_DIR, filename)


@app.route("/background-image")
def background_image():
    """
    Serves the background image used by `templates/base.html`.
    """
    if not (BACKGROUND_ASSET_DIR / BACKGROUND_ASSET_FILENAME).exists():
        abort(404)
    return send_from_directory(BACKGROUND_ASSET_DIR, BACKGROUND_ASSET_FILENAME)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Serverless imports do not execute the __main__ block, so initialize the local
# SQLite schema before serving requests. On Vercel this database is temporary;
# use a managed database for persistent production data.
init_db()

if __name__ == "__main__":
    app.run(debug=True)
