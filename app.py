import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# =====================
# LOAD MODEL
# =====================

model = YOLO(
    r"C:\Users\admin\runs\detect\train-6\weights\best.pt"
)

# =====================
# CONFIG
# =====================

st.set_page_config(
    page_title="Nhận diện biển báo giao thông",
    page_icon="🚦",
    layout="wide"
)

# =====================
# CSS
# =====================

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}

.header {
    background: linear-gradient(135deg, #0f172a, #1d4ed8);
    padding: 18px 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 18px;
}

.header h1 {
    font-size: 28px;
    margin: 0;
}

.header p {
    margin-top: 6px;
    font-size: 14px;
    color: #dbeafe;
}

.result-card {
    background: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
}

.result-card h3 {
    margin-top: 0;
    color: #065f46;
    font-size: 20px;
}

.result-card p {
    font-size: 15px;
    margin-bottom: 8px;
}

.metric-box {
    background: #dbeafe;
    color: #1e40af;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: bold;
    display: inline-block;
}

.image-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.upload-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

div[data-testid="stImage"] img {
    max-height: 360px;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# =====================
# CLASS DATA
# =====================

CLASS_VN = {
    "limit40": "Biển giới hạn tốc độ 40 km/h",
    "limit60": "Biển giới hạn tốc độ 60 km/h",
    "limit80": "Biển giới hạn tốc độ 80 km/h",
    "limit100": "Biển giới hạn tốc độ 100 km/h",
    "limit120": "Biển giới hạn tốc độ 120 km/h",
    "not_go_straight": "Biển cấm đi thẳng",
    "not_turn_left": "Biển cấm rẽ trái",
    "not_turn_right": "Biển cấm rẽ phải",
    "not_u_turn_left": "Biển cấm quay đầu bên trái",
    "not_u_turn_right": "Biển cấm quay đầu bên phải"
}

SIGN_INFO = {
    "limit40": "Phương tiện không được chạy quá tốc độ 40 km/h.",
    "limit60": "Phương tiện không được chạy quá tốc độ 60 km/h.",
    "limit80": "Phương tiện không được chạy quá tốc độ 80 km/h.",
    "limit100": "Phương tiện không được chạy quá tốc độ 100 km/h.",
    "limit120": "Phương tiện không được chạy quá tốc độ 120 km/h.",
    "not_go_straight": "Cấm các phương tiện đi thẳng.",
    "not_turn_left": "Cấm các phương tiện rẽ trái.",
    "not_turn_right": "Cấm các phương tiện rẽ phải.",
    "not_u_turn_left": "Cấm các phương tiện quay đầu sang trái.",
    "not_u_turn_right": "Cấm các phương tiện quay đầu sang phải."
}

# =====================
# HEADER
# =====================

st.markdown("""
<div class="header">
    <h1>🚦 Hệ thống nhận diện biển báo giao thông</h1>
    <p>Ứng dụng YOLOv8 phát hiện và phân loại biển báo giao thông từ hình ảnh tải lên</p>
</div>
""", unsafe_allow_html=True)

# =====================
# UPLOAD
# =====================

st.markdown('<div class="upload-title">📤 Tải ảnh lên</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Chọn ảnh biển báo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =====================
# MAIN
# =====================

if uploaded_file is None:
    st.info("Vui lòng tải ảnh biển báo lên để bắt đầu nhận diện.")

else:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]
    result_img = result.plot()

    col1, col2, col3 = st.columns([1, 1, 0.9], gap="small")

    with col1:
        st.markdown('<div class="image-title">🖼️ Ảnh đầu vào</div>', unsafe_allow_html=True)
        st.image(
            image,
            channels="BGR",
            use_container_width=True
        )

    with col2:
        st.markdown('<div class="image-title">🤖 Ảnh sau nhận diện</div>', unsafe_allow_html=True)
        st.image(
            result_img,
            channels="BGR",
            use_container_width=True
        )

    with col3:
        st.markdown('<div class="image-title">📌 Kết quả</div>', unsafe_allow_html=True)

        if len(result.boxes) == 0:
            st.warning("Không phát hiện biển báo.")
        else:
            for i, box in enumerate(result.boxes, start=1):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]

                label_vn = CLASS_VN.get(label, label)
                description = SIGN_INFO.get(label, "Chưa có mô tả.")

                st.markdown(f"""
                <div class="result-card">
                    <h3>Biển báo {i}</h3>
                    <p><b>Loại:</b> {label_vn}</p>
                    <p><b>Mô tả:</b> {description}</p>
                    <div class="metric-box">Độ tin cậy: {confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)