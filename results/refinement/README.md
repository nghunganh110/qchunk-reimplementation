# Thử nghiệm mở rộng: tinh chỉnh action chunk bằng gradient Q

Các checkpoint lớn không được lưu trong Git. Để chạy lại các lệnh bên dưới,
đặt checkpoint puzzle cuối tại `runs/puzzle_3x3_task4/offline_final.msgpack`
và các checkpoint trung gian trong
`runs/puzzle_3x3_task4/checkpoints/`. Các hash trong tệp JSON giúp kiểm tra
đúng phiên bản checkpoint.

## Ý tưởng và cách chạy

Sau khi QC chọn chunk tốt nhất trong `N` mẫu từ flow, chỉ ở lúc suy luận, cập nhật chunk hai lần theo gradient của **trung bình hai critic**. Mỗi bước có độ dài L2 `step_size`, chặn hành động vào `[-1, 1]`, rồi giới hạn khoảng cách L2 so với chunk ban đầu đã chặn bởi `radius`. Biến thể hình học lấy ba chunk flow gần chunk được QC chọn nhất theo khoảng cách Euclid, rồi chiếu gradient Q vào không gian do ba vector hiệu tạo ra bằng phép chiếu ridge. Nó chỉ thay hướng cập nhật; số mẫu, bước, độ dài bước và bán kính giữ như gradient nhỏ. Đây là một xấp xỉ hình học dựa trên vài mẫu, **không bảo đảm** chunk tinh chỉnh vẫn thuộc phân phối flow. Không thay checkpoint, flow, critic hoặc thuật toán huấn luyện. Mã: `src/qchunk/refinement.py`; lệnh đánh giá: `src/qchunk/evaluate_refinement.py`.

Các cấu hình được thử: QC `N=8`; `N=8` + gradient nhỏ (`step_size=0.05`, `radius=0.1`); `N=8` + gradient lớn (`step_size=0.25`, `radius=0.5`); `N=8` + gradient hình học (`step_size=0.05`, `radius=0.1`, 3 láng giềng); QC gốc `N=32` cho cube và `N=64` cho puzzle. Kết quả gradient lớn trên cube được quan sát trước khi chọn cấu hình nhỏ; biến thể hình học được đề xuất sau khi xem các lần chạy đó. Vì đều dùng lại 50 seed đánh giá, các kết quả này là **thăm dò hậu nghiệm**, chưa phải kiểm định trên tập đánh giá độc lập.

Đánh giá cùng checkpoint, 50 episode, seed đánh giá 42, CPU. Mỗi variant dùng lại cùng seed reset episode và cùng chuỗi khóa PRNG; `N=8` với/không gradient chọn từ đúng các mẫu flow giống nhau. `N=32/64` cũng dùng cùng seed nhưng số mẫu khác nên tập ứng viên không giống `N=8`. Thời gian biên dịch JIT được tách khỏi thời gian rollout. Mỗi episode được lưu trong JSON để đối chiếu theo cặp. Đây là một seed **huấn luyện** cho mỗi task; 50 episode đánh giá không thay thế nhiều seed huấn luyện.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.evaluate_refinement \
  --config configs/cube_triple.yaml \
  --checkpoint runs/cube_triple/online_final.msgpack \
  --output results/refinement/cube_eval_50.json --episodes 50 \
  --variants qc_n8 qc_n8_grad qc_n32

JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.evaluate_refinement \
  --config configs/cube_triple.yaml \
  --checkpoint runs/cube_triple/online_final.msgpack \
  --output results/refinement/cube_small_radius_50.json --episodes 50 \
  --variants qc_n8_grad_small --diagnostics-episodes 10

JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.evaluate_refinement \
  --config configs/puzzle_3x3.yaml \
  --checkpoint runs/puzzle_3x3_task4/offline_final.msgpack \
  --output results/refinement/puzzle_eval_50.json --episodes 50 \
  --variants qc_n8 qc_n8_grad_small qc_n64

JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.evaluate_refinement \
  --config configs/cube_triple.yaml \
  --checkpoint runs/cube_triple/online_final.msgpack \
  --output results/refinement/cube_geometric_50.json --episodes 50 \
  --variants qc_n8_grad_geo

JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.evaluate_refinement \
  --config configs/puzzle_3x3.yaml \
  --checkpoint runs/puzzle_3x3_task4/offline_final.msgpack \
  --output results/refinement/puzzle_geometric_50.json --episodes 50 \
  --variants qc_n8_grad_geo
```

## Kết quả

| Task | Cách chọn chunk | Thành công | Độ dài TB | Thời gian rollout (s) | ms/chunk trong rollout |
| --- | --- | ---: | ---: | ---: | ---: |
| Cube task2 | QC N=8 | 47/50 | 262,56 | 37,42 | 14,14 |
| Cube task2 | N=8 + gradient, r=0,1 | 47/50 | 215,98 | 32,33 | 14,83 |
| Cube task2 | N=8 + gradient hình học, r=0,1 | 47/50 | 233,92 | 34,42 | 14,60 |
| Cube task2 | N=8 + gradient, r=0,5 | 44/50 | 281,76 | 41,04 | 14,48 |
| Cube task2 | QC N=32 | 49/50 | 184,86 | 43,84 | 23,40 |
| Puzzle task4 | QC N=8 | 50/50 | 285,84 | 44,71 | 15,52 |
| Puzzle task4 | N=8 + gradient, r=0,1 | 49/50 | 268,54 | 44,40 | 16,42 |
| Puzzle task4 | N=8 + gradient hình học, r=0,1 | 50/50 | 272,40 | 40,38 | 14,72 |
| Puzzle task4 | QC N=64 | 50/50 | 228,78 | 59,63 | 25,82 |

`ms/chunk` lấy tổng thời gian rollout chia số chunk được chọn; số này gồm cả bước môi trường và chi phí Python, **không** phải latency thuần của mạng. Tổng thời gian rollout phụ thuộc số bước môi trường mỗi policy tạo ra, nên không dùng nó làm phép so sánh tốc độ suy luận độc lập.

So với QC N=8, gradient nhỏ rút ngắn trung bình 46,58 bước trên cube và 17,30 bước trên puzzle, nhưng khoảng bootstrap 95% theo **cặp episode** lần lượt là `[-124,4; 28,5]` và `[-38,6; 4,7]`: cả hai vẫn chứa 0. Tỷ lệ thành công không tăng: cube giữ 47/50, puzzle giảm một episode. Gradient lớn trên cube làm giảm 3 lần thành công và tăng độ dài trung bình. Không nên kết luận cấu hình nhỏ cải thiện ổn định từ các số liệu này.

Gradient hình học giữ 47/50 trên cube và 50/50 trên puzzle, giống QC N=8. So với QC N=8, độ dài trung bình giảm 28,64 bước trên cube và 13,44 bước trên puzzle. Khoảng bootstrap 95% theo cặp lần lượt là `[-107,2; 49,1]` và `[-31,8; 5,8]`, đều chứa 0. So với gradient nhỏ không chiếu, cube dài hơn 17,94 bước và puzzle dài hơn 3,86 bước; trên puzzle chỉ hơn một lần thành công. Phép chiếu **chưa chứng minh được** lợi ích ổn định về độ dài hay tỷ lệ thành công. Cả ba hướng dùng cùng checkpoint và seed đánh giá; các khoảng bootstrap chỉ mô tả biến thiên giữa episode trong lần huấn luyện này, không đo biến thiên giữa các seed huấn luyện.

Kiểm tra cơ chế trên 128 trạng thái lấy đều từ 490 chunk của 10 episode do QC N=8 chạy trên cube: với `r=0.5`, Q trung bình hai critic tăng trung bình `3,76`, ở `96,9%` trạng thái được lấy mẫu. Khoảng cách L2 trung bình của chunk là `0,373`, tối đa `0,499` (ràng buộc `r=0,5`). Cả hai critic đều tăng Q trung bình, nhưng thành công thật trên 50 episode giảm từ 47/50 xuống 44/50. Điều này phù hợp với nguy cơ tối ưu theo critic ở vùng hành động kém tin cậy; **chưa đủ** để chứng minh đó là nguyên nhân duy nhất của các lần thất bại.

## Tệp kết quả

- `cube_pilot_5.json`: pilot đầu tiên; không dùng để khẳng định chất lượng.
- `cube_eval_50.json`: QC N=8, gradient lớn, QC N=32.
- `cube_small_radius_50.json`: gradient nhỏ và phép chẩn đoán Q trên trạng thái QC N=8 đã gặp.
- `puzzle_eval_50.json`: QC N=8, gradient nhỏ, QC N=64.
- `cube_geometric_50.json`: gradient hình học trên cube, 50 episode.
- `puzzle_geometric_50.json`: gradient hình học trên puzzle, 50 episode.

## Kiểm tra thêm trên checkpoint puzzle giai đoạn sớm

Các checkpoint offline trung gian của cùng lần huấn luyện puzzle được lưu trong thư mục kết quả Kaggle do người thực hiện cung cấp. Vì checkpoint cuối đã đạt gần trần thành công, em dùng **20 episode sàng lọc với seed đánh giá 1000** để chọn mốc chưa chạm trần, rồi **50 episode mới với seed 2026** để so sánh các policy. Đây là thay seed đánh giá và checkpoint của **cùng một lần huấn luyện**, không phải huấn luyện thêm seed.

| Checkpoint offline | QC N=8, 20 episode | QC N=64, 20 episode | Quyết định |
| --- | ---: | ---: | --- |
| 50.000 cập nhật | 12/20 | 2/20 | Chọn để đánh giá chính |
| 100.000 cập nhật | 20/20 | 20/20 | Đã chạm trần trong pilot |

Không có checkpoint 75.000 bước trong thư mục kết quả (tệp `offline_step_0750000.msgpack` là **750.000** bước). Bốn tệp `puzzle_50k_screen20.json`, `puzzle_50k_n8_screen20.json`, `puzzle_100k_screen20.json`, và `puzzle_100k_n8_screen20.json` lưu phép sàng lọc. Checkpoint 50.000 bước có SHA-256 `c350e7a68432...`; hash đầy đủ trong JSON. Kết quả chính ở `puzzle_50k_holdout50.json`.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.evaluate_refinement \
  --config configs/puzzle_3x3.yaml \
  --checkpoint runs/puzzle_3x3_task4/checkpoints/offline_step_0050000.msgpack \
  --output results/refinement/puzzle_50k_holdout50.json \
  --episodes 50 --seed 2026 \
  --variants qc_n8 qc_n8_grad_small qc_n8_grad_geo qc_n64
```

| Policy tại 50.000 bước | Thành công trên 50 episode mới | Độ dài TB | Thời gian rollout (s) | ms/chunk trong rollout |
| --- | ---: | ---: | ---: | ---: |
| QC N=8 | 31/50 (62%) | 440,74 | 55,01 | 12,45 |
| N=8 + gradient nhỏ | 27/50 (54%) | 446,88 | 59,89 | 13,36 |
| N=8 + gradient hình học | 28/50 (56%) | 449,28 | 59,91 | 13,30 |
| QC N=64 | 13/50 (26%) | 480,40 | 119,13 | 24,77 |

So với QC N=8, gradient thường đổi 12 episode thất bại thành công nhưng cũng đổi 16 episode thành công thành thất bại; gradient hình học tương ứng là 9 và 12. Chênh lệch tỷ lệ thành công theo cặp episode là `-8 điểm %` (bootstrap 95% `[-28; 12]`) và `-6 điểm %` (`[-24; 12]`). Chưa có bằng chứng hai bước gradient cải thiện policy tại checkpoint sớm.

QC N=64 chỉ đạt 13/50 so với 31/50 của N=8. Nó mất 24 episode mà N=8 giải được và được thêm 6 episode mà N=8 thất bại; chênh lệch là `-36 điểm %` với bootstrap 95% theo cặp `[-54; -16]`. Đây là kết quả trên **một checkpoint**, không khẳng định N=64 luôn kém hơn N=8 hoặc chứng minh nguyên nhân là critic đánh giá quá cao. Lúc 100.000 bước, cả N=8 và N=64 đều đạt 20/20 ở phép sàng lọc, cho thấy sự khác biệt quan sát được gắn với giai đoạn học sớm. Các policy dùng cùng seed reset episode; N=8 và N=64 có số mẫu khác nhau nên không nhận chính xác cùng tập chunk ứng viên.

## Chẩn đoán Q và thử quy tắc chọn N theo checkpoint

Để kiểm tra giả thuyết trên, `src/qchunk/diagnose_q_selection.py` ghi Q của **hai critic** tại mỗi chunk được chọn và return chiết khấu thực nhận từ thời điểm đó đến hết episode. Q được lấy từ chính chunk do policy QC gốc chọn; phép ghi số liệu không thay hành động. Với chunk đầu tiên của episode, đặt `G_0 = sum_t gamma^t r_t` đến lúc kết thúc hoặc hết giới hạn thời gian, `e_N = Q_mean(s_0,a_0^N) - G_0`. `e_N` là **phần dư so với return hữu hạn quan sát được**, không phải sai số Q theo giá trị thật đã biết. Script còn ghi chênh lệch giữa hai critic và độ cách biệt Q của chunk thắng so với Q trung bình các ứng viên tại trạng thái đầu.

Mỗi policy dùng cùng seed reset cho từng episode; khóa PRNG của policy được tạo riêng bằng `fold_in(PRNGKey(seed), episode_index)`. Giao thức này khác script đánh giá gradient trước đó, nên chỉ đối chiếu số lượt thành công trong cùng bộ chẩn đoán. Hai checkpoint là của **cùng một lần huấn luyện**; mỗi hàng dưới đây có 50 episode, seed đánh giá 3000.

| Checkpoint | N | Thành công | Q đầu TB | G đầu TB | e đầu TB | Khoảng thắng Q đầu TB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50.000 | 8 | 30/50 | -79,339 | -98,287 | 18,948 | 1,196 |
| 50.000 | 64 | 11/50 | -78,109 | -99,001 | 20,891 | 2,403 |
| 100.000 | 8 | 47/50 | -82,537 | -96,592 | 14,055 | 0,662 |
| 100.000 | 64 | 50/50 | -81,982 | -94,380 | 12,398 | 1,225 |

Ở 50.000 bước, tăng N từ 8 lên 64 nâng Q đầu trung bình `1,230` nhưng G đầu giảm `0,714`; phần dư tăng `1,944` và số thành công giảm 19/50. Bootstrap ghép cặp theo episode (20.000 lần lấy lại mẫu) cho chênh lệch phần dư `+[1,419; 2,486]` và chênh lệch thành công `[-56; -20]` điểm phần trăm. Khoảng thắng Q cũng tăng từ `1,196` lên `2,403`. Đây là mẫu hình **phù hợp** với chọn ứng viên có Q lạc quan ở critic sớm. Tại 100.000 bước, N64 tăng G đầu `2,212` và **giảm** phần dư `1,657`, đồng thời đạt 50/50 thay vì 47/50. Chênh lệch giữa hai critic không tách rõ hai trường hợp: tại 50.000 bước, giá trị trung bình của nó là `2,213` cho N8 và `2,039` cho N64; hai critic có thể cùng sai theo một hướng.

Trước khi xem seed 4000, em cố định quy tắc ở **cấp checkpoint**: tính `D = mean(e_64) - mean(e_8)` trên bộ hiệu chuẩn; chọn N8 nếu `D > 0`, ngược lại chọn N64. Bộ seed 3000 cho `D=+1,944` ở 50.000 bước và `D=-1,657` ở 100.000 bước, nên lần lượt chọn N8 và N64. Đánh giá độc lập 30 episode/checkpoint với seed 4000 cho kết quả:

| Checkpoint | N8 cố định | N64 cố định | N được chọn | Thành công của quy tắc |
| --- | ---: | ---: | ---: | ---: |
| 50.000 | 19/30 | 5/30 | 8 | 19/30 |
| 100.000 | 26/30 | 30/30 | 64 | 30/30 |
| Gộp hai mốc | 45/60 | 35/60 | 8 rồi 64 | 49/60 |

Độ dài trung bình khi gộp 60 episode là `368,15` bước với quy tắc, `400,27` với N8 và `390,30` với N64. Chênh lệch 49/60 so với 45/60 là **mô tả của đúng hai checkpoint này**, chưa đủ khẳng định một thuật toán tổng quát. Hiệu chuẩn cần 50 episode cho **mỗi N ở mỗi checkpoint** (200 episode), lớn hơn bộ validation 120 episode; vì vậy đây là một phép chẩn đoán và chọn cấu hình khi có quyền chạy môi trường, chưa phải cải tiến suy luận rẻ. G hữu hạn bị ảnh hưởng bởi cắt episode ở 500 bước; N8 cũng khác N64 dùng trong đích huấn luyện. Các yếu tố đó và sai số chung của hai critic ngăn việc xem `e_N` là thước đo trực tiếp độ lệch critic hay một bằng chứng nhân quả.

Tệp kết quả: `puzzle_50k_q_trace50.json`, `puzzle_100k_q_trace50.json` (hiệu chuẩn); `puzzle_50k_q_validation30.json`, `puzzle_100k_q_validation30.json` (validation). Mỗi JSON chứa hash checkpoint, seed, kết quả từng episode, Q từng chunk và return chiết khấu.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m qchunk.diagnose_q_selection \
  --config configs/puzzle_3x3.yaml \
  --checkpoint runs/puzzle_3x3_task4/checkpoints/offline_step_0050000.msgpack \
  --output results/refinement/puzzle_50k_q_trace50.json \
  --episodes 50 --seed 3000 --candidates 8 64
```

Thay checkpoint 50.000 bằng `offline_step_0100000.msgpack` để chạy mốc 100.000; thay `--episodes 50 --seed 3000` bằng `--episodes 30 --seed 4000` cho bộ validation.
