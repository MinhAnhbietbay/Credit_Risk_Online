# Task 10 — Cơ sở chọn ngưỡng

**Quyết định:** ngưỡng chính = chặn 21.9% hồ sơ điểm cao nhất — đúng tỉ lệ Home Credit đã từ chối trên 1,327,459 hồ sơ lịch sử (`previous_application`); trên OOF ensemble = 0.117 (P 0.217, R 0.588). Đối chiếu: KS/Youden 0.082 (P 0.178, R 0.718).

**So sánh:** tỉ lệ từ chối lịch sử (chung, theo loại) / KS / chi phí theo số tiền thật với LGD 0.3–1.0 / **Số liệu:** chi phí thật cho ngưỡng 0.28–0.58 tùy LGD — quá nhạy với một tham số không có trong dữ liệu (proxy LGD≈0.36 chỉ n=48) nên không dùng làm ngưỡng chính; ```
                          basis      kind  threshold  flag_rate  precision  recall    f1
          từ chối lịch sử 21.9%     CHÍNH      0.117      0.219      0.217   0.588 0.317
       từ chối Cash loans 34.7% tham khảo      0.076      0.347      0.172   0.738 0.279
   từ chối Consumer loans 10.7% tham khảo      0.190      0.107      0.290   0.384 0.330
  từ chối Revolving loans 33.6% tham khảo      0.079      0.336      0.175   0.728 0.282
                      KS/Youden ĐỐI CHIẾU      0.082      0.325      0.178   0.718 0.286
         chi phí thật, LGD=0.30       phụ      0.580      0.001      0.643   0.008 0.016
         chi phí thật, LGD=0.45       phụ      0.470      0.006      0.569   0.039 0.073
         chi phí thật, LGD=0.60       phụ      0.400      0.013      0.504   0.079 0.137
         chi phí thật, LGD=0.80       phụ      0.330      0.027      0.428   0.144 0.215
         chi phí thật, LGD=1.00       phụ      0.280      0.045      0.381   0.212 0.273
                 cũ: fn=5, fp=1        bỏ      0.150      0.158      0.249   0.486 0.329
```
