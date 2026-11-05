# Đối chiếu quy mô nhỏ với kết quả QC công bố

## Nguồn và cách dựng lại

- Bài báo: Q. Li, Z. Zhou, S. Levine, *Reinforcement Learning with Action Chunking*, NeurIPS 2025, <https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html>.
- Dữ liệu biểu đồ theo từng tác vụ do tác giả phát hành: <https://github.com/ColinQiyangLi/qc/blob/main/plot_data/README.md>, tệp `ogbench-individual.pkl`. Bản được dùng ở đây tải từ <https://raw.githubusercontent.com/ColinQiyangLi/qc/main/plot_data/ogbench-individual.pkl>; SHA-256: `c3151eb06cb1a3d1f1efcd59f58e1fdb4bde9588134b3cca9c11dc6536fa5709`.
- `paper_qc_reference.csv` chỉ trích ra đường QC của `cube-triple-play-task2` và `puzzle-3x3-play-task4` từ tệp trên. Bước 0–1 triệu là huấn luyện offline; 1–2 triệu là pha online.
- Số liệu cài đặt lại lấy từ `results/modal_cube_triple/learning_curve_evaluation.csv` và `results/puzzle_3x3_task4/metrics.csv`. Hai manifest tương ứng lưu cấu hình và môi trường của lần chạy. Tệp metrics/manifest của puzzle được sao chép từ thư mục kết quả người thực hiện cung cấp; checkpoint vẫn ở thư mục gốc của lần chạy.
- Chạy `MPLCONFIGDIR=/tmp/mpl-qchunk-repro python tools/compare_paper_qc.py` tại gốc repo để dựng `paper_vs_reimplementation.png`.
- Kiểm tra thêm tính tái chạy của checkpoint: nạp lại `offline_final.msgpack` của puzzle và đánh giá 50 episode với seed 42 trên CPU, không huấn luyện. Kết quả là 50/50, return trung bình −227,78 và độ dài trung bình 228,78; lệnh, hash checkpoint và phiên bản thư viện ở `puzzle_cpu_reevaluation.json`. Log GPU ban đầu cũng là 50/50, với return −231,12 và độ dài 232,12. Hai lần có cùng tỷ lệ thành công nhưng không trùng quỹ đạo hoàn toàn; khác backend và phiên bản JAX, nên không quy nguyên nhân chênh lệch cho một yếu tố cụ thể.
- Checkpoint online cube được tải từ Modal Volume `qchunk-experiments` về `runs/cube_triple/online_final.msgpack` (không đưa tệp 35 MB vào Git). Đánh giá lại 50 episode trên CPU, cùng seed 42, đạt 49/50 = 98%, return trung bình −351,20 và độ dài 184,86; log GPU gốc là 48/50 = 96%, return −387,62, độ dài 209,16. Lệnh và hash checkpoint ở `cube_cpu_reevaluation.json`. Hai lần đều cho thấy mô hình sau online giải được hầu hết episode; chênh một episode không có ý nghĩa thống kê với một seed huấn luyện. Backend và phiên bản JAX khác nhau, nên không khẳng định nguyên nhân chênh lệch là gì.

## Mức độ khớp của giao thức

Hai lần chạy dùng cùng tác vụ OGBench với paper, `H=5`, batch 256, hai critic, 10 bước flow, mạng 4 lớp ẩn rộng 512, 1 triệu cập nhật offline và 50 episode mỗi lần đánh giá. `N=32` với cube-triple và `N=64` với puzzle-3x3, đúng theo bảng siêu tham số QC của bài báo. Puzzle dùng phần thưởng thưa. Cube chạy thêm 1 triệu bước môi trường online; puzzle mới chạy pilot 10.000 bước online, nên chỉ mốc sau offline của puzzle là đối chiếu ngang bước với bài báo.

| Tác vụ, mốc | Paper: trung bình 4 seed | Cài đặt lại: 1 seed, 50 episode |
| --- | ---: | ---: |
| Cube-triple task2, sau 1M offline | 0% | 0/50 = 0% |
| Cube-triple task2, sau 1M online | 89% (KTC 95%: 81,5–93,5%) | 48/50 = 96% |
| Puzzle-3x3 task4, sau 1M offline | 100% | 50/50 = 100% |
| Puzzle-3x3 task4, sau 10k online | Không có mốc tương ứng | 50/50 = 100% |

Đường cube-triple cũng cùng xu hướng: gần 0% sau offline và tăng mạnh trong pha online. Tại 300.000 bước online, paper đạt 36,5% còn cài đặt lại đạt 36%; các mốc khác không trùng khít. Bài báo lấy trung bình **4 seed huấn luyện**, với khoảng tin cậy bootstrap 95%; cài đặt lại dùng **1 seed huấn luyện** để tái hiện ở quy mô nhỏ. Khoảng tin cậy quanh trung bình của paper không phải khoảng dự báo cho mọi seed riêng lẻ. Do đó, 96% so với 89% không chứng minh cài đặt lại tốt hơn. Kết quả hiện có đủ để trình bày việc tái hiện **xu hướng và các mốc chính trên hai task**; chỉ cần thêm seed hoặc baseline nếu đặt mục tiêu đánh giá độ ổn định hay so sánh hơn kém giữa các thuật toán.

Không đưa pilot puzzle 10.000 bước ngang với mốc 100.000 bước online gần nhất trong dữ liệu tác giả. Cũng không lấy 100% trên một seed làm bằng chứng rằng mọi seed sẽ đạt 100%.
