# SGKR & SPHR — Hệ thống khuyến nghị sản phẩm điện tử

**Tiếng Việt** · [English](README.en.md)

Bài tập lớn cá nhân môn *Học máy nâng cao*. Tôi đề xuất hai mô hình gợi ý top-N: **SGKR** (mô hình item–item giải trực tiếp
bằng công thức) và **SPHR** (mô hình dựa trên lịch sử mua, huấn luyện bằng gradient), rồi so sánh với năm
baseline đã công bố trên bộ dữ liệu Amazon Electronics. Mọi mô hình dùng chung dữ liệu, cách chia, bộ đánh giá
xếp hạng trên toàn catalog và các chỉ số; mọi con số trong tài liệu này đều tính lại được từ các file kết quả
đã lưu.

## Điểm chính

- **SGKR có NDCG@20 cao nhất ở mật độ đầy đủ (0,0383)** và tốt hơn LightGCN, XSimGCL, DirectAU một cách có
  ý nghĩa thống kê (p sau hiệu chỉnh Holm = 0,0006), trong khi chỉ mất **2,5 phút trên CPU** để huấn luyện,
  nhanh hơn khoảng 17 lần so với mạng nơ-ron đồ thị nhanh nhất.
- **SPHR mạnh nhất khi dữ liệu thưa:** NDCG@20 cao nhất ở 25 % và 50 % số tương tác train.
- **So với các mô hình giải trực tiếp cùng kiểu**, SGKR ngang EASE đầy đủ ở mật độ đầy đủ (EASE cần ma trận
  7,5 GB, không lưu được), và tốt hơn bản EASE rút gọn về cỡ lưu được cũng như GF-CF.
- **Hạn chế:** SGKR không phải tốt nhất ở mật độ 50 %, và mọi kết quả đến từ một seed duy nhất trên tập
  validation (xem [Hạn chế](#hạn-chế)).

## Mục lục

[Tái lập kết quả](#tái-lập-kết-quả) · [Demo](#demo) · [Mô hình](#mô-hình) · [Dữ liệu](#dữ-liệu) ·
[Kết quả](#kết-quả) · [Giao thức đánh giá](#giao-thức-đánh-giá) · [Hạn chế](#hạn-chế) ·
[Cấu trúc thư mục](#cấu-trúc-thư-mục) · [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Tái lập kết quả

```bash
conda create -n AML python=3.11 && conda activate AML
pip install -r requirements.txt

python scripts/prepare_data.py                         # tải từ SNAP và dựng bộ dữ liệu
python scripts/audit_data.py                           # kiểm tra dữ liệu

python scripts/train_model.py --model sgkr --density 1.0
#   --model    sgkr | sphr | lightgcn | xsimgcl | directau | ease | gfcf
#   --density  1.0 | 0.75 | 0.5 | 0.25

python scripts/summarize_results.py                    # results/summary.csv
python scripts/make_report.py                          # hình H1–H8, bảng B1–B4 (so sánh chính)
python scripts/make_closed_form_report.py              # hình closed-form, bảng B5–B6 (so sánh bổ sung)
python scripts/verify_results.py                       # tính lại mọi chỉ số từ các file đã lưu
python -m pytest -q tests
```

Mỗi run ghi vào `results/<model>/d<mật độ>/seed42/`: `metrics.json`, dự đoán validation, checkpoint,
`history.csv` (với mô hình huấn luyện theo epoch) và `logs/` (kế hoạch run, bản lưu mã nguồn, tiến độ, ngân sách
tài nguyên). Run đã xong được bỏ qua. Các mô hình huấn luyện lặp chạy đủ 200 epoch, không dừng sớm, và giữ
checkpoint có NDCG@20 validation cao nhất. `python scripts/run_experiments.py` chạy thực nghiệm chính: SGKR, SPHR
và 3 GNN ở 4 mật độ.

**Phần cứng.** SGKR và GF-CF chạy được trên CPU laptop. SPHR và các baseline đồ thị được train trên GPU Kaggle
(`kaggle/kaggle_train.ipynb`). EASE phải nghịch đảo ma trận dày 44.843 × 44.843 nên cần **khoảng 9,1 GB RAM
trống**; mô hình này được chạy trên Kaggle CPU (`kaggle/kaggle_ease.ipynb`).

## Demo

```bash
python -m streamlit run app/streamlit_app.py
```

Giao diện tiếng Việt, bố cục cho màn hình 1920 × 1080.

- **Gợi ý cho khách:** top-10 của cả 7 mô hình đặt cạnh nhau, đánh dấu sản phẩm khách mua tiếp theo, kèm phần
  giải thích của SGKR cho gợi ý số 1.
- **Khách mới:** tự tạo lịch sử bằng cách tìm sản phẩm. SGKR, SPHR, EASE và GF-CF gợi ý ngay; các baseline đồ
  thị không phục vụ được user chưa có lúc train.
- **Đánh giá mô hình:** bảng xếp hạng, biểu đồ, kiểm định thống kê và hình báo cáo, chia thành nhóm cần huấn
  luyện và nhóm giải trực tiếp.

Demo nạp các checkpoint đã lưu và chỉ đọc tập train, validation.

## Mô hình

| Mô hình | Họ | Cách tạo ra | Phần cứng |
|---|---|---|---|
| **SGKR** (đề xuất) | Hồi quy item–item | Giải trực tiếp bằng công thức | CPU |
| **SPHR** (đề xuất) | Phân rã dựa trên lịch sử | Adam, softmax trên toàn catalog | GPU |
| LightGCN [1] | Mạng nơ-ron đồ thị | Huấn luyện gradient, tối đa 200 epoch | GPU |
| XSimGCL [2] | Mạng nơ-ron đồ thị + học tương phản | Huấn luyện gradient, tối đa 200 epoch | GPU |
| DirectAU [3] | Mục tiêu alignment–uniformity | Huấn luyện gradient, tối đa 200 epoch | GPU |
| EASE [4] | Hồi quy item–item | Giải trực tiếp bằng công thức | CPU |
| GF-CF [5] | Lọc trên đồ thị | Giải trực tiếp (SVD) | CPU |

**SGKR** là hồi quy ridge item–item có chuẩn hóa theo đồ thị. Mỗi sản phẩm được dự đoán từ 128 sản phẩm liên
quan nhất, với mức chuẩn hóa thích nghi; hệ phương trình được giải chính xác (Cholesky) và loại chính sản phẩm
mục tiêu khỏi ngữ cảnh của nó. SGKR không có tham số riêng cho từng user và không cần bộ tối ưu, nên chấm điểm
được ngay cho bất kỳ lịch sử mua nào, kể cả user chưa từng xuất hiện khi train. Mô hình cũng giải thích được
mỗi gợi ý bằng các sản phẩm trong lịch sử đóng góp nhiều nhất.

**SPHR** biểu diễn user bằng lịch sử mua đã chuẩn hóa trên các bảng item được lọc khuếch tán:

```math
\operatorname{score}(u, i) = \mathbf{q}_u^{\top} \tilde{\mathbf{v}}_i + b_i,
\qquad \mathbf{q}_u = |H_u|^{-1/2} \sum_{j \in H_u} \tilde{\mathbf{s}}_j
```

trong đó $`H_u`$ là lịch sử mua của user, $`\tilde{\mathbf{s}}_j`$ và $`\tilde{\mathbf{v}}_i`$ là vector nguồn và vector đích
đã lọc của sản phẩm, $`b_i`$ là độ lệch riêng của sản phẩm. Mô hình được huấn luyện bằng Adam với softmax trên toàn
catalog theo ngữ cảnh leave-one-out. Giống SGKR, SPHR phục vụ được user mới chỉ từ lịch sử.

Ba baseline đồ thị học một vector riêng cho từng user lúc train, nên không gợi ý được cho user mới nếu không
huấn luyện lại.

## Dữ liệu

Đánh giá Amazon Electronics 5-core [6, 7]. Đánh giá từ 4 sao trở lên là tương tác dương; user và sản phẩm có ít
nhất 5 tương tác; lịch sử của mỗi user được chia theo thời gian.

| Thống kê | Giá trị |
|---|---:|
| Số user | 124.895 |
| Số sản phẩm | 44.843 |
| Tương tác train | 858.192 |
| Tương tác validation | 107.274 |
| Độ thưa tập train | 99,98 % |
| Độ dài lịch sử, trung bình / trung vị | 6,87 / 5 |

Để đo độ bền khi dữ liệu khan hiếm, mọi mô hình còn được train trên **75 %, 50 % và 25 %** số tương tác train
(lấy mẫu với seed 42).

## Kết quả

### Độ chính xác

NDCG@20 trên tập validation: xếp hạng trên toàn bộ catalog, loại các sản phẩm user đã mua, 107.215 user được
đánh giá.

| Mô hình | 25 % | 50 % | 75 % | 100 % | Thời gian train ở 100 % |
|---|---:|---:|---:|---:|---:|
| **SGKR** | 0,0080 | 0,0171 | 0,0293 | **0,0383** | 2,5 phút (CPU) |
| **SPHR** | **0,0177** | **0,0227** | 0,0290 | 0,0352 | 127 phút (GPU) |
| XSimGCL | 0,0112 | 0,0193 | 0,0276 | 0,0329 | 63 phút (GPU) |
| DirectAU | 0,0103 | 0,0205 | 0,0270 | 0,0321 | 85 phút (GPU) |
| LightGCN | 0,0084 | 0,0184 | 0,0226 | 0,0265 | 43 phút (GPU) |
| EASE | 0,0067 | 0,0192 | **0,0294** | 0,0380 | 11 phút (CPU) |
| GF-CF | 0,0039 | 0,0038 | 0,0084 | 0,0189 | 45 phút (CPU) |

Thời gian train đo trên phần cứng khác nhau (CPU và GPU), chỉ nên so theo bậc độ lớn.

<p align="center">
  <img src="reports/figures/H1_density_ndcg20.png" width="49%" alt="NDCG@20 theo mật độ, nhóm cần huấn luyện">
  <img src="reports/figures/H1_closed_form_density_ndcg20.png" width="49%" alt="NDCG@20 theo mật độ, nhóm giải trực tiếp">
</p>
<p align="center"><sub>NDCG@20 theo mật độ dữ liệu train. Trái: SGKR, SPHR và các baseline đồ thị. Phải: SGKR và
các baseline giải trực tiếp.</sub></p>

### So sánh thống kê

Các phép so sánh chia thành hai nhóm, mỗi nhóm hiệu chỉnh Holm riêng. Mọi kiểm định là paired bootstrap theo
user (10.000 lần lấy mẫu), một phía theo hướng mô hình đề xuất tốt hơn.

**So sánh chính: mô hình đề xuất so với mạng nơ-ron đồ thị** (bảng B3, mật độ đầy đủ).

| So sánh | Δ NDCG@20 | KTC 95 % | p Holm |
|---|---:|---|---:|
| SGKR − LightGCN | +0,0117 | [+0,0109; +0,0126] | 0,0006 |
| SGKR − XSimGCL | +0,0054 | [+0,0046; +0,0061] | 0,0006 |
| SGKR − DirectAU | +0,0062 | [+0,0055; +0,0070] | 0,0006 |
| SPHR − LightGCN | +0,0086 | [+0,0080; +0,0093] | 0,0006 |
| SPHR − XSimGCL | +0,0023 | [+0,0017; +0,0028] | 0,0006 |
| SPHR − DirectAU | +0,0031 | [+0,0025; +0,0038] | 0,0006 |

**So sánh bổ sung: SGKR với các mô hình giải trực tiếp** (bảng B5–B6, Holm trên 3 phép so sánh mỗi mật độ).
Nhóm so sánh này được lập sau khi đã xem kết quả nên mang tính khám phá.

| SGKR − … | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|
| EASE (đầy đủ, 7,5 GB) | SGKR hơn | EASE hơn | ngang (p = 0,72) | ngang (p = 0,051) |
| EASE top-128 (25–32 MB) | SGKR hơn | EASE hơn | SGKR hơn | SGKR hơn |
| GF-CF | SGKR hơn | SGKR hơn | SGKR hơn | SGKR hơn |

Checkpoint SGKR chiếm 6–41 MB tùy mật độ, cỡ tương đương EASE top-128.

**Chỗ SGKR không thắng.** Ở mật độ 50 %, SGKR thấp hơn cả 3 baseline đồ thị và cả hai bản EASE. Ở 25 % và 50 %,
SPHR là mô hình mạnh nhất. GF-CF yếu trên dữ liệu rất thưa này vì cách chuẩn hóa bậc đối xứng của nó ưu tiên
các sản phẩm ít người mua.

## Giao thức đánh giá

- **Mục tiêu:** sản phẩm mua tiếp theo của mỗi user. Mọi mô hình xếp hạng toàn bộ 44.843 sản phẩm, loại các
  sản phẩm user đã mua, và được đánh giá trên cùng một nhóm user.
- **Chỉ số:** NDCG@10/20, Recall@10/20 và MRR@20.
- **Chọn cấu hình:** chỉ so trên một nửa số user validation (chia theo băm ID); nửa còn lại dành để xác nhận.
  Mọi quyết định được ghi vào nhật ký theo kiểu chỉ thêm.
- **Kiểm định:** paired bootstrap theo user, một phía, hiệu chỉnh Holm trong từng nhóm so sánh.
- **Cấu hình baseline:** EASE và GF-CF mỗi mô hình dùng một cấu hình cố định, không tinh chỉnh trên dữ liệu
  này: EASE λ = 300 (giá trị giữa lưới thường dùng 100/300/1000), GF-CF theo mặc định của tác giả (k = 256,
  α = 0,3). Demo dùng EASE rút gọn còn 128 trọng số lớn nhất mỗi sản phẩm; mọi số EASE ở trên là ma trận đầy đủ
  trừ khi ghi *top-128*.

## Hạn chế

- **Một seed (42).** Chưa đo độ dao động giữa các seed; khoảng tin cậy chỉ phản ánh việc lấy mẫu user.
- **Chỉ trên validation.** Tập test được giữ riêng; không con số nào ở đây là kết quả test cuối cùng.
- **So sánh bổ sung mang tính khám phá.** Nhóm so sánh với mô hình giải trực tiếp được thêm sau khi đã biết kết
  quả.
- **Baseline giải trực tiếp chưa tinh chỉnh.** EASE và GF-CF không được tinh chỉnh trên dữ liệu này, nên điểm
  của chúng có thể thấp hơn mức tốt nhất chúng đạt được.
- **Một bộ dữ liệu.** Mọi kết luận chỉ áp dụng cho Amazon Electronics.

## Cấu trúc thư mục

```
configs/   cấu hình đã khóa: sgkr, sphr, lightgcn, xsimgcl, directau, ease, gfcf, common
src/       xử lý dữ liệu, bộ đánh giá, hàm mất mát, vòng huấn luyện, các mô hình (src/models/*.py)
scripts/   các lệnh chạy; scripts/verify/ chứa kiểm chứng SymPy cho các nghiệm đóng
app/       demo Streamlit: streamlit_app.py (giao diện), recommender.py (nạp mô hình), benchmark.py (thống kê)
kaggle/    notebook dùng trên Kaggle
results/   mỗi run một thư mục, cùng summary.csv (git chỉ lưu checkpoint và lịch sử huấn luyện)
reports/   figures/ (ảnh PNG) và tables/ (bảng CSV)
tests/     bộ kiểm thử pytest
```

## Tài liệu tham khảo

1. X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, M. Wang. *LightGCN: Simplifying and Powering Graph Convolution
   Network for Recommendation.* SIGIR 2020.
2. J. Yu, X. Xia, T. Chen, L. Cui, N. Q. V. Hung, H. Yin. *XSimGCL: Towards Extremely Simple Graph Contrastive
   Learning for Recommendation.* IEEE TKDE 2023.
3. C. Wang, Y. Yu, W. Ma, M. Zhang, C. Chen, Y. Liu, S. Ma. *Towards Representation Alignment and Uniformity in
   Collaborative Filtering.* KDD 2022.
4. H. Steck. *Embarrassingly Shallow Autoencoders for Sparse Data.* WWW 2019.
5. Y. Shen, Y. Wu, Y. Zhang, C. Shan, J. Zhang, K. B. Letaief, D. Li. *How Powerful is Graph Convolution for
   Recommendation?* CIKM 2021.
6. R. He, J. McAuley. *Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class
   Collaborative Filtering.* WWW 2016.
7. J. McAuley, C. Targett, Q. Shi, A. van den Hengel. *Image-Based Recommendations on Styles and Substitutes.*
   SIGIR 2015.
