# Qwen Edit Studio

Giao diện web gọn nhẹ cho workflow **sửa ảnh Qwen Image 2.1** của ComfyUI (template `image_qwen_image_2_1_image_edit`), chạy ngay trong ComfyUI tại `http://127.0.0.1:8188/qwenedit`.

## Tính năng

- Ô soạn prompt hỗ trợ thẻ `<image1>` / `@image_1`, tự đánh lại số khi bạn đổi thứ tự ảnh
- Tối đa 10 ảnh: ảnh 1 là ảnh được sửa, các ảnh còn lại là ảnh tham chiếu. Kéo thả để tải lên, dán từ clipboard, đổi thứ tự, rê chuột để xem ảnh phóng to
- Kích thước được đặt riêng cho từng ảnh: ảnh 1 theo thiết lập kích thước; ảnh tham chiếu lớn hơn giới hạn (mặc định ~1 MP; có các mức ~1,5, ~2,4, ~4 MP hoặc không giới hạn) sẽ được thu nhỏ về mức đó (mỗi ảnh một node `ImageScale`, bước resize của encoder được tắt). Trang hiển thị dung lượng file → kích thước đưa vào model của từng ảnh, kích thước ảnh đầu ra, và cảnh báo khi ảnh 1 vượt quá 2K
- Thiết lập: ngân sách kích thước ảnh 1 (giữ nguyên / 1 MP / 2,4 MP / 2K), khung ảnh đầu ra (theo ảnh 1 hoặc tự nhập rộng × cao), steps, CFG, sampler, scheduler, seed, 1–4 ảnh mỗi lần bấm
- Tùy chọn bộ cải thiện prompt Qwen 3.5 (viết lại câu lệnh trước khi đưa vào model sửa ảnh); prompt đã viết lại được hiện trong chi tiết job
- Chọn model diffusion, text encoder, VAE và bộ cải thiện prompt, cùng các thiết lập prefix cache của Qwen Image 2.1. Khi chọn model diffusion, trang tự chọn text encoder và VAE đã cài có cùng độ chính xác (`int8_convrot`, `bf16`…); Qwen Image 2.1 chỉ có một VAE bf16, dùng chung cho mọi model
- Bảng kết quả có tiến độ trực tiếp, thanh trượt so sánh trước/sau, hộp chi tiết (thời gian, ảnh đầu vào, prompt) và nút "Sửa tiếp" một chạm (dùng thẳng file trong `ComfyUI/output`, không sao chép). Có thể ẩn kết quả khỏi danh sách, hoặc xoá hẳn cùng file ảnh (sau khi xác nhận)
- Liên kết trên header tới [H3 Studio](https://github.com/anonymoux/H3_Studio) (`/h3studio`), ứng dụng tạo video MiniMax H3 chạy cùng ComfyUI
- Banner liệt kê các file model mà thiết lập hiện tại cần nhưng chưa cài, kèm link tải
- Bản nháp và danh sách job được lưu trên server (`user/qwenedit_state.json`) và đẩy tới các trang đang mở qua websocket của ComfyUI, nên script và trang web dùng chung một trạng thái

## Cài đặt

Chép thư mục này vào `ComfyUI/custom_nodes/` rồi khởi động lại ComfyUI. Cần bản ComfyUI có các node `TextEncodeQwenImage21`, `QwenImage21Cache` và `TextGenerate` (0.43 trở lên).

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/anonymoux/Qwen-image-2.1-Edit-Shared.git
```

Workflow được dựng trong `builder.py` (subgraph của template đã được làm phẳng). `pe_system_prompt.md` là system prompt của bộ cải thiện prompt, lấy nguyên từ template.

## Model

Đường dẫn tính từ `ComfyUI/models/`. Nguồn: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

### Bắt buộc (~16 GB)

| Thư mục | File | Dung lượng |
|---|---|---|
| `diffusion_models/` | [qwen_image_2.1_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors) | 6.8 GB |
| `text_encoders/` | [qwen3vl_8b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors) | 8.7 GB |
| `vae/` | [qwen_image_2.1_vae_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors) | 0.6 GB |

### Tùy chọn

| Tính năng | Thư mục | File | Dung lượng |
|---|---|---|---|
| Bộ cải thiện prompt | `text_encoders/` | [qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors) | 8.8 GB |
| Model độ chính xác đầy đủ | `diffusion_models/` | [qwen_image_2.1_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_bf16.safetensors) | 13.3 GB |
| Text encoder độ chính xác đầy đủ | `text_encoders/` | [qwen3vl_8b_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_bf16.safetensors) | 16.3 GB |

## API

```
GET    /qwenedit/options            danh sách model, lựa chọn sampler/scheduler/cache, model còn thiếu
GET    /qwenedit/state              {rev, draft, jobs}
PUT    /qwenedit/state              {draft: {...một phần}}
POST   /qwenedit/queue              đưa bản nháp hiện tại vào hàng đợi (draft.count job)
POST   /qwenedit/jobs/{id}/cancel   huỷ job
POST   /qwenedit/jobs/{id}/delete   xoá job cùng file ảnh kết quả
DELETE /qwenedit/jobs/{id}          ẩn khỏi danh sách (giữ file)
```

## Mẹo (từ template)

- Giữ CFG = 1 theo cách chạy chính thức; negative prompt chỉ có tác dụng khi CFG > 1.
- Pipeline chính thức dùng khoảng 40–50 steps với euler; mặc định ở đây là 25.
- Bật `thinking` khi dùng bộ cải thiện prompt: bộ này được huấn luyện kèm khối suy luận.
