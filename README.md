# So sánh mô hình gợi ý trên Amazon Electronics

Đồ án môn Học máy nâng cao. Dự án đề xuất 2 mô hình, **SGKR** và **SPHR**, và so sánh chúng với
3 baseline đồ thị đã công bố (**LightGCN**, **XSimGCL**, **DirectAU**). Cả 5 mô hình dùng cùng dữ liệu,
cùng cách chia leave-one-out, cùng bộ đánh giá xếp hạng trên toàn bộ catalog và cùng thiết lập
(d = 64, Adam lr 3e-4, batch 2048, tối đa 100 epoch, early stopping theo NDCG@20 validation, seed 42).

## Kết quả hiện tại (chỉ validation; tập test chưa được đọc)

| Mô hình | Loại | NDCG@20 (validation) |
|---|---|---:|
| **SGKR** | Đề xuất: ridge regression item–item, nghiệm đóng | **0.03829** |
| **SPHR** (arm D2) | Đề xuất: phân rã theo lịch sử user, softmax, Adam | **0.03546** |
| XSimGCL | Baseline | 0.03289 |
| DirectAU | Baseline | 0.03191 |
| LightGCN | Baseline | 0.02525 |

Số liệu được tái lập bằng `scripts/verify_results.py`. Lần dùng nửa confirmation thứ 3 và phần chấm
test cuối cùng (Cổng E) vẫn đang chờ.

## Cấu trúc thư mục

```
configs/
  sgkr.yaml                  cấu hình SGKR đã khóa (đọc bởi scripts/run_sgkr.py)
  sphr.yaml                  cấu hình SPHR = arm D2 (đọc bởi scripts/tune_sphr.py)
  lightgcn.yaml xsimgcl.yaml directau.yaml common.yaml   cấu hình 3 baseline và phần dùng chung
src/
  data/ evaluation/ losses/ training/ utils/   dữ liệu, bộ đánh giá, vòng train dùng chung
  models/                    5 mô hình được báo cáo và các thành phần của chúng
    lightgcn.py xsimgcl.py directau.py         3 baseline
    sgkr.py                                    SGKR
    sphr.py                                    SPHR
scripts/                     các lệnh chính (xem bên dưới)
  verify/                    các kiểm chứng toán học/lý thuyết (sympy, ví dụ nhỏ; không train)
  data_tools/                công cụ bảo trì bộ dữ liệu (metadata, rollback, kiểm sạch)
  extra/                     ItemKNN, RP3β, EASE (dùng khi chấm lại checkpoint)
tests/                       pytest
results/                     kết quả: <model>/d<mật độ>/seed<seed>/, summary.csv, archive/ (không đưa lên git)
reports/                     hình (.png) và bảng (.csv) sinh bởi make_report.py
app/                         demo: streamlit_app.py (giao diện), recommender.py (nạp mô hình, xếp hạng)
```

## Lệnh chạy (CMD trong VS Code, tại `D:\advanced-machine`, môi trường `AML`)

Dữ liệu:
```cmd
python scripts\prepare_data.py
python scripts\audit_data.py
```

Train một mô hình (một lệnh chung cho cả 5; `--model lightgcn|xsimgcl|directau|sgkr|sphr`,
`--density 1.0|0.75|0.5|0.25`):
```cmd
python scripts\train_model.py --model xsimgcl --density 1.0 --seed 42 --epochs 200
```
Mỗi run ghi vào `results\<model>\d<mật độ %>\seed<seed>\`: `metrics.json` (chỉ số, epoch tốt nhất, thời gian,
cấu hình, môi trường `env`: GPU, torch, git commit), `val_predictions.npy`, `val_users.npy`, `history.csv`
(cùng cột cho mọi mô hình: `epoch, loss, NDCG@10, NDCG@20, Recall@10, Recall@20, seconds, learning_rate`,
sau đó các cột riêng `extra_*`), checkpoint và `logs\`. Run đã xong (có `metrics.json`) không chạy lại.
Lệnh này gọi các runner gốc (`train_suite.py`, `run_sgkr.py`, `tune_sphr.py`). Không dừng sớm: mọi mô hình
train đủ `--epochs` (mặc định 200) và giữ checkpoint có NDCG@20 validation cao nhất.

Train toàn bộ thực nghiệm lần lượt (5 mô hình × 4 mật độ), bỏ qua run đã xong, rồi tự vẽ hình:
```cmd
python scripts\run_experiments.py --dry-run
python scripts\run_experiments.py
```
Có thể giới hạn: `--densities 0.75 0.5`, `--models sgkr sphr`, `--seeds 42 2024`; nhật ký ở
`results\run_experiments.log`.

Hình và bảng cho báo cáo (chỉ ảnh `.png`, cùng khung 1920×1200; bảng .csv trong `reports\tables\`):
```cmd
python scripts\summarize_results.py
python scripts\make_report.py
```
`make_report.py` in ra các run (mô hình × mật độ) còn thiếu kèm lệnh chạy. Kết quả cũ trước khi thống nhất
bố cục nằm trong `results\archive\`.

So sánh theo cặp trên validation (paired bootstrap, Holm):
```cmd
python scripts\compare_val_runs.py RUN_THAM_CHIEU RUN_B --half selection --holm --output results\compare\NEW.json
```

Tái lập kết quả của cả 5 mô hình từ các file đã lưu (không train, không đọc test):
```cmd
python scripts\verify_results.py --output results\verify\report.json
```

Demo (sau khi đã train; mô hình nào chưa có checkpoint ở mật độ đang chọn thì hiện "Chưa train"):
```cmd
streamlit run app\streamlit_app.py
```
Ba tab: **User có sẵn** (top-10 của 5 mô hình cạnh nhau, ✓ khi trúng sản phẩm mua tiếp theo, giải thích của SGKR),
**User mới** (tự chọn sản phẩm; SGKR và SPHR gợi ý ngay, baseline cần train lại), **Kết quả** (hình và bảng trong
`reports\`). Chỉ đọc train và validation. User mẫu chọn theo quy tắc cố định: lịch sử và đáp án đủ tên, hãng;
3 user cho mỗi nhóm độ dài lịch sử (3–4, 5–9, ≥10), seed 42.

Kiểm thử:
```cmd
python -m pytest -q tests
```
`tests/test_smoke_five_models.py` chạy lệnh thật của cả 5 mô hình trên dữ liệu tổng hợp nhỏ (CPU, vài giây).

Smoke test trên dữ liệu thật (mỗi mô hình 50 epoch, không dừng sớm, SGKR fit đầy đủ; mặc định đủ 4 mật độ
1.0, 0.75, 0.5, 0.25; `--epochs 1` để kiểm tra nhanh):
```cmd
python scripts\smoke_test.py
python scripts\smoke_test.py --density 1.0 --models sgkr sphr
```

## Giao thức đánh giá

- Validation được chia theo băm user ID thành **nửa selection** (chọn cấu hình) và **nửa confirmation**
  (chỉ dùng để xác nhận, mỗi lần đều ghi vào nhật ký). Tập test chỉ được đọc một lần, ở Cổng E.
- So sánh dùng paired bootstrap trên từng user, hiệu chỉnh Holm; mức chênh tối thiểu phát hiện được
  (MDE) khoảng 0.0005 NDCG@20.
- Mọi thay đổi kế hoạch được ghi theo kiểu chỉ thêm (append-only) trong nhật ký nghiên cứu.
