# Watermark Remover

Xóa watermark từ ảnh tự động.

## 🚀 Cách Dùng

**3 Bước:**

1. **Chạy:** `run.bat`

2. **Chọn ảnh** → Click button "Chọn File" hoặc "Chọn Thư Mục"

3. **Xóa:**
   - **Tự động**: Click "XÓA WATERMARK" (xóa góc trên trái)
   - **Chọn vùng**: Chọn radio "Chọn bằng chuột" → Vẽ khung trên ảnh → Click "XÓA WATERMARK"

4. **Lưu:** Click "Lưu" → ảnh lưu vào `output/`

---

## 📁 Cấu Trúc

```
water MARK/
├── watermark_remover.py    # App chính
├── run.bat                 # Chạy app
├── input/                  # Đặt ảnh vào đây (optional)
└── output/                 # Kết quả lưu ở đây
```

---

## ⚙️ Tính Năng

- ✅ Xóa tự động (góc trên trái)
- ✅ Chọn vùng bằng chuột
- ✅ Xử lý hàng loạt
- ✅ Auto lưu vào `output/`

---

## 🔧 Cài Đặt

```bash
pip install -r requirements.txt
```

Hoặc chạy `run.bat` - tự động cài.

---

Made with Python + OpenCV
