import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import pyscreenshot as ImageGrab
import pyautogui
import time
import os
import random
import json
import threading
import numpy as np
import cv2 # OpenCV for image processing
import xml.etree.ElementTree as ET # For Pascal VOC XML parsing

# --- PyTorch 관련 라이브러리 (조건부 임포트) ---
try:
    import torch
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import functional as TF
    from torchvision import transforms as T # Import T for common transforms
    
    # Custom Dataset class for object detection
    class CustomObjectDataset(Dataset):
        def __init__(self, root, transforms=None):
            self.root = root
            # If no transforms are provided, default to converting to tensor
            # This ensures that images returned are always PyTorch tensors
            self.transforms = transforms if transforms is not None else T.ToTensor() 
            self.imgs = [f for f in sorted(os.listdir(os.path.join(root, "images"))) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.labels = [f for f in sorted(os.listdir(os.path.join(root, "labels"))) if f.endswith('.xml')]
            
            if len(self.imgs) != len(self.labels):
                print(f"Warning: Image count ({len(self.imgs)}) does not match label count ({len(self.labels)}).")
                # Attempt to filter out unmatched files if names don't align
                img_names = {os.path.splitext(f)[0] for f in self.imgs}
                label_names = {os.path.splitext(f)[0] for f in self.labels}
                common_names = sorted(list(img_names.intersection(label_names)))
                self.imgs = [f"{name}.png" for name in common_names] # Assuming all saved as .png
                self.labels = [f"{name}.xml" for name in common_names]
                if len(self.imgs) == 0:
                     raise ValueError("No matching image and label files found after filtering.")


        def __getitem__(self, idx):
            img_name = self.imgs[idx]
            label_name = os.path.splitext(img_name)[0] + ".xml" # Assume same base name, different extension
            
            img_path = os.path.join(self.root, "images", img_name)
            label_path = os.path.join(self.root, "labels", label_name)

            img = Image.open(img_path).convert("RGB")
            
            # Parse XML label file to get bounding box
            boxes = []
            labels = []
            try:
                tree = ET.parse(label_path)
                root_xml = tree.getroot() 
                
                for obj in root_xml.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = max(0, int(float(bndbox.find('xmin').text)))
                    ymin = max(0, int(float(bndbox.find('ymin').text)))
                    xmax = min(img.width, int(float(bndbox.find('xmax').text)))
                    ymax = min(img.height, int(float(bndbox.find('ymax').text)))
                    
                    if xmax <= xmin or ymax <= ymin: # Skip invalid boxes
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1) # Class ID for your captured object (0 is background)

            except Exception as e:
                print(f"Error parsing XML for {label_path}: {e}")
                boxes = []
                labels = []

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            # Apply transforms: Now self.transforms will always convert to tensor
            # If you add more complex transforms (e.g., random horizontal flip),
            # they need to be applied to both image and target (bounding boxes)
            # using a custom transform logic (e.g., from Albumentations or custom class).
            img = self.transforms(img) 

            return img, target

        def __len__(self):
            return len(self.imgs)

    # Collate function for DataLoader when batching lists of targets
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Basic transforms for inference (ToTensor only) - now just for clarity, can be removed if not used elsewhere.
    # The default transform in CustomObjectDataset now handles this.
    class ToTensorTransform:
        def __call__(self, image): # Removed target as T.ToTensor() only takes image
            image = TF.to_tensor(image)
            return image
            
except ImportError:
    torch = None
    torchvision = None
    FasterRCNN_ResNet50_FPN_Weights = None
    FastRCNNPredictor = None
    DataLoader = None
    Dataset = None
    CustomObjectDataset = None
    collate_fn = None
    ToTensorTransform = None
    print("PyTorch and/or torchvision not found. Deep learning features will be disabled.")


# --- Faster R-CNN 모델 및 함수 관리 클래스 ---
class FasterRCNNManager:
    def __init__(self, app_instance):
        self.app = app_instance # AutomationApp 인스턴스 참조
        self.saved_functions = self.load_functions()
        self.current_loaded_model = None 
        
        # CUDA (GPU) 사용 가능 여부 확인 및 장치 설정
        if torch is not None and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device('cpu')
            print("Using device: CPU")

    def load_functions(self):
        """저장된 이미지 탐색 함수 목록을 불러옵니다."""
        if os.path.exists("saved_faster_rcnn_functions.json"):
            with open("saved_faster_rcnn_functions.json", "r", encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_functions(self):
        """현재 함수 목록을 저장합니다."""
        with open("saved_faster_rcnn_functions.json", "w", encoding='utf-8') as f:
                json.dump(self.saved_functions, f, indent=4, ensure_ascii=False)

    def _generate_training_data_variations(self, original_image_path, func_name, num_variations=100):
        """
        원본 이미지를 기반으로 다양한 학습 데이터 변형을 생성합니다.
        실제 Faster R-CNN 학습을 위한 이미지와 바운딩 박스 라벨을 생성해야 합니다.
        """
        output_image_dir = os.path.join("faster_rcnn_dataset", "images")
        output_label_dir = os.path.join("faster_rcnn_dataset", "labels")
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        try:
            original_image = Image.open(original_image_path).convert("RGB")
            # 캡처된 이미지의 바운딩 박스는 (0,0,width,height)로 간주합니다.
            original_bbox = [0, 0, original_image.width, original_image.height]

            # 메시지박스 대신 UI 상태 업데이트
            self.app.update_training_status("데이터 생성 중...", 0)

            for i in range(num_variations):
                img = original_image.copy()
                bbox = list(original_bbox) # [xmin, ymin, ymax, ymax]

                # --- 이미지 및 바운딩 박스 변형 ---
                # 1. 크기 조절 (Scale)
                scale_factor = random.uniform(0.8, 1.2) # 80% ~ 120%
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                # 바운딩 박스도 동일하게 스케일
                bbox = [int(b * scale_factor) for b in bbox]

                # 2. 랜덤 회전 (Rotation) - 바운딩 박스 변환이 복잡하여 단순화
                if random.random() < 0.3: # 30% 확률로 회전
                    angle = random.randint(-10, 10) # 작은 각도
                    img = img.rotate(angle, expand=True, fillcolor=(0,0,0)) # 검은색으로 빈 공간 채움
                    # 회전 후에는 바운딩 박스가 왜곡되므로, 여기서는 임시로 이미지 전체를 재설정합니다.
                    # 실제 프로젝트에서는 Bounding Box-aware augmentation 라이브러리 (e.g., Albumentations) 사용.
                    bbox = [0, 0, img.width, img.height]


                # 3. 색상, 밝기, 대비 조절 (Color Jitter)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))

                # 4. 노이즈 추가 (Noise)
                if random.random() < 0.2: # 20% 확률로 노이즈
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
                
                # 5. 저장
                image_filename_base = f"{func_name}_{i}"
                image_filename = f"{image_filename_base}.png"
                label_filename = f"{image_filename_base}.xml"

                img.save(os.path.join(output_image_dir, image_filename))

                # --- Pascal VOC XML 라벨 생성 ---
                root_elem = ET.Element("annotation")
                ET.SubElement(root_elem, "folder").text = "images"
                ET.SubElement(root_elem, "filename").text = image_filename
                ET.SubElement(root_elem, "path").text = os.path.abspath(os.path.join(output_image_dir, image_filename))

                size_elem = ET.SubElement(root_elem, "size")
                ET.SubElement(size_elem, "width").text = str(img.width)
                ET.SubElement(size_elem, "height").text = str(img.height)
                ET.SubElement(size_elem, "depth").text = "3" # RGB 이미지

                obj_elem = ET.SubElement(root_elem, "object")
                ET.SubElement(obj_elem, "name").text = func_name # 객체 이름 = 함수 이름
                ET.SubElement(obj_elem, "pose").text = "Unspecified"
                ET.SubElement(obj_elem, "truncated").text = "0"
                ET.SubElement(obj_elem, "difficult").text = "0"

                bndbox_elem = ET.SubElement(obj_elem, "bndbox")
                # 바운딩 박스 좌표가 이미지 경계를 벗어나지 않도록 클리핑
                ET.SubElement(bndbox_elem, "xmin").text = str(max(0, bbox[0]))
                ET.SubElement(bndbox_elem, "ymin").text = str(max(0, bbox[1]))
                ET.SubElement(bndbox_elem, "xmax").text = str(min(img.width, bbox[2]))
                ET.SubElement(bndbox_elem, "ymax").text = str(min(img.height, bbox[3]))

                tree = ET.ElementTree(root_elem)
                tree.write(os.path.join(output_label_dir, label_filename), encoding="utf-8", xml_declaration=True)
                
                self.app.update_training_status(f"데이터 생성 중... ({i+1}/{num_variations})", int((i+1)/num_variations * 100))

            self.app.update_training_status(f"'{num_variations}'장의 학습 데이터 생성 완료.", 100)
            # messagebox.showinfo("데이터 생성 완료", f"'{num_variations}'장의 학습 데이터가 생성되었습니다.")
            return True
        except Exception as e:
            self.app.update_training_status(f"데이터 생성 오류: {e}", 0)
            messagebox.showerror("데이터 생성 오류", f"학습 데이터 생성 중 오류 발생: {e}")
            return False

    def train_object_detection_model(self, original_image_path, func_name):
        """
        Faster R-CNN 모델 학습을 시작합니다.
        PyTorch 코드를 완성하여 실제 학습이 이루어집니다.
        """
        if torch is None or torchvision is None:
            messagebox.showerror("오류", "PyTorch 및 torchvision이 설치되어 있지 않습니다. 모델 학습을 위해 설치해주세요.")
            self.app.update_training_status("PyTorch/torchvision 미설치", 0)
            return

        # 데이터 생성 함수를 스레드 외부에서 호출하여 완료를 기다리게 함
        # 데이터 생성 중에도 UI가 업데이트되도록 하기 위함
        if not self._generate_training_data_variations(original_image_path, func_name):
            return

        self.app.update_training_status(f"'{func_name}' 함수를 위한 Faster R-CNN 모델 학습 시작...", 0)

        try:
            # --- Training setup ---
            # CustomObjectDataset now defaults to T.ToTensor() if no transforms are provided
            dataset = CustomObjectDataset(root="faster_rcnn_dataset") 
            
            # 데이터 로더는 배치 처리를 위해 collate_fn을 사용
            data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

            # Load pre-trained model on COCO and replace the classifier
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            
            # num_classes = 1 (captured_object) + 1 (background) = 2
            num_classes = 2 
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 모델을 설정된 장치(CPU 또는 GPU)로 이동
            model.to(self.device)

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

            num_epochs = 20 # 작은 데이터셋이므로 과적합 방지를 위해 에폭을 너무 높이지 않음

            print("Starting Faster R-CNN Training...")
            for epoch in range(num_epochs):
                model.train() # Set model to training mode
                epoch_loss = 0
                for i, (images, targets) in enumerate(data_loader):
                    # Ensure targets are valid (contain at least one box for non-background images)
                    valid_batch = []
                    for img, tgt in zip(images, targets):
                        # Ensure images are PyTorch tensors here
                        if tgt["boxes"].numel() > 0: # Check if there are any bounding boxes
                            valid_batch.append((img, tgt))
                        else:
                            print(f"Skipping image {tgt['image_id']} with no valid bounding boxes.")
                    
                    if not valid_batch:
                        continue # Skip current batch if no valid images

                    # 이미지를 설정된 장치(CPU 또는 GPU)로 이동
                    images_valid = [item[0].to(self.device) for item in valid_batch]
                    # 타겟(바운딩 박스, 라벨 등)을 설정된 장치(CPU 또는 GPU)로 이동
                    targets_valid = [{k: v.to(self.device) for k, v in t.items()} for t in [item[1] for item in valid_batch]]

                    loss_dict = model(images_valid, targets_valid)
                    losses = sum(loss for loss in loss_dict.values())
                    epoch_loss += losses.item()

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                
                lr_scheduler.step()
                
                # UI 업데이트 (에폭별 손실)
                progress_percent = int(((epoch + 1) / num_epochs) * 100)
                self.app.update_training_status(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}", progress_percent)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

            # 모델 저장
            model_save_dir = "models"
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f"{func_name}_faster_rcnn_model.pth")
            torch.save(model.state_dict(), model_path)
            
            self.app.update_training_status(f"'{func_name}' 모델 학습 완료 및 저장되었습니다.", 100)
            messagebox.showinfo("학습 완료", f"'{func_name}' 모델 학습 완료 및 저장되었습니다.\n경로: {model_path}")

            self.saved_functions[func_name] = {"type": "image_search", "model_path": model_path}
            self.save_functions()

        except Exception as e:
            self.app.update_training_status(f"학습 오류 발생: {e}", 0)
            messagebox.showerror("학습 오류", f"모델 학습 중 예상치 못한 오류 발생: {e}")


    def load_model_for_inference(self, model_path):
        """
        주어진 경로에서 Faster R-CNN 모델을 로드합니다.
        """
        if torch is None or torchvision is None:
            if not self.app.is_sequence_running: # 순서 실행 중이 아닐 때만 팝업
                messagebox.showerror("오류", "PyTorch 및 torchvision이 설치되어 있지 않아 모델을 로드할 수 없습니다.")
            return None

        try:
            # 모델 구조 로드 (weights=None으로 가중치 없이)
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            num_classes = 2 # 학습 시 사용한 클래스 수와 동일
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 학습된 가중치 로드. map_location을 지정하여 CPU/GPU에 관계없이 로드 가능
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval() # 평가 모드 설정 (추론 시 필수)
            
            if not self.app.is_sequence_running: # 순서 실행 중이 아닐 때만 팝업
                messagebox.showinfo("모델 로드", f"모델 로드 완료: {model_path}")
            return model
        except Exception as e:
            if not self.app.is_sequence_running: # 순서 실행 중이 아닐 때만 팝업
                messagebox.showerror("모델 로드 오류", f"모델 로드 중 오류 발생: {e}")
            return None

    def find_image_and_move_mouse(self, func_name, confidence_threshold=0.7):
        """
        학습된 Faster R-CNN 모델을 사용하여 화면에서 이미지를 탐색하고
        마우스를 이동시키는 로직을 포함합니다.
        """
        if torch is None or torchvision is None:
            if not self.app.is_sequence_running:
                messagebox.showerror("오류", "PyTorch 및 torchvision이 설치되어 있지 않아 이미지 탐지가 불가능합니다.")
            return False

        if func_name not in self.saved_functions:
            if not self.app.is_sequence_running:
                messagebox.showerror("오류", f"'{func_name}' 함수를 찾을 수 없습니다. 먼저 학습시켜야 합니다.")
            return False

        model_info = self.saved_functions[func_name]
        if model_info["type"] == "image_search":
            model_path = model_info["model_path"]

            # 모델 로드 (필요 시에만 로드)
            if self.current_loaded_model is None or \
               (isinstance(self.current_loaded_model, str) and self.current_loaded_model != model_path) or \
               (hasattr(self.current_loaded_model, 'state_dict') and str(self.current_loaded_model.state_dict()) != str(torch.load(model_path, map_location=self.device))): 
                self.current_loaded_model = self.load_model_for_inference(model_path)
                if self.current_loaded_model is None:
                    return False

            if not self.app.is_sequence_running:
                messagebox.showinfo("탐색 중", f"'{func_name}' 이미지 화면에서 탐색 중... (Faster R-CNN)")

            try:
                # 1. 현재 화면 캡처
                screen_screenshot = ImageGrab.grab()
                
                # PIL Image를 PyTorch 텐서로 변환
                # ToTensorTransform now only takes image
                input_tensor = T.ToTensor()(screen_screenshot) 
                # 입력 텐서를 설정된 장치(CPU 또는 GPU)로 이동
                input_tensor = input_tensor.unsqueeze(0).to(self.device) # 배치 차원 추가 및 디바이스 이동

                # 2. 모델 추론 (Inference)
                with torch.no_grad(): # 추론 시에는 gradient 계산 비활성화
                    predictions = self.current_loaded_model(input_tensor)
                
                # 3. 예측 결과 파싱 및 후처리
                # 결과 텐서를 CPU로 이동하여 numpy로 변환
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()

                best_detection = None
                max_score = -1

                # 클래스 1 (캡처된 객체) 중 가장 높은 점수 찾기
                for i in range(len(scores)):
                    if labels[i] == 1 and scores[i] > confidence_threshold:
                        if scores[i] > max_score:
                            max_score = scores[i]
                            best_detection = boxes[i]

                if best_detection is not None:
                    min_x, min_y, max_x, max_y = best_detection
                    
                    width = max_x - min_x
                    height = max_y - min_y

                    rand_x = int(min_x + random.uniform(0, width))
                    rand_y = int(min_y + random.uniform(0, height))
                    
                    # 순서 실행 중이 아닐 때만 팝업
                    if not self.app.is_sequence_running:
                        messagebox.showinfo("탐색 성공", f"'{func_name}' 이미지 발견! (신뢰도: {max_score:.2f})\n마우스 이동: ({rand_x}, {rand_y})")
                    pyautogui.moveTo(rand_x, rand_y, duration=0.2)
                    return True
                else:
                    # 순서 실행 중이 아닐 때만 팝업
                    if not self.app.is_sequence_running:
                        messagebox.showwarning("탐색 실패", f"'{func_name}' 이미지를 화면에서 찾을 수 없습니다. (최고 신뢰도: {max_score:.2f})")
                    return False

            except Exception as e:
                # 순서 실행 중이 아닐 때만 팝업
                if not self.app.is_sequence_running:
                    messagebox.showerror("탐지 오류", f"이미지 탐지 중 예상치 못한 오류 발생: {e}")
                return False
        return False

    # --- 새로운 마우스 이동 함수 ---
    def _move_mouse_absolute(self, x, y):
        """마우스를 주어진 절대 좌표로 이동시킵니다."""
        try:
            pyautogui.moveTo(x, y, duration=0.2)
            # 순서 실행 중이 아닐 때만 팝업
            if not self.app.is_sequence_running:
                messagebox.showinfo("마우스 이동", f"마우스를 ({x}, {y})로 이동했습니다.")
            return True
        except Exception as e:
            # 순서 실행 중이 아닐 때만 팝업
            if not self.app.is_sequence_running:
                messagebox.showerror("마우스 이동 오류", f"마우스 이동 중 오류 발생: {e}")
            return False

# --- 메인 애플리케이션 클래스 ---
class AutomationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("자동화 프로그램")
        self.root.geometry("320x400") # 창 크기 약간 증가
        self.root.resizable(False, False)

        self.is_sequence_running = False # 순서 실행 중인지 여부를 나타내는 플래그
        self.object_detector_manager = FasterRCNNManager(self) # FasterRCNNManager에 self (AutomationApp 인스턴스) 전달
        self.capture_area = None
        self.captured_image_tk = None
        self.current_captured_image_path = None
        self.training_status_label = None # 학습 진행 상태를 표시할 라벨
        self.training_progress_bar = None # 학습 진행 바 (텍스트)

        self.order_functions = []
        
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=8)
        self.root.grid_rowconfigure(0, weight=1)

        self.category_frame = tk.Frame(self.root, bg="lightgray", bd=2, relief="groove")
        self.category_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.content_frame = tk.Frame(self.root, bg="white", bd=2, relief="groove")
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self.create_category_buttons()
        self.show_default_content()

        self.mouse_x, self.mouse_y = 0, 0
        self.is_capturing = False
        self.capture_window = None

    def create_category_buttons(self):
        btn_image_capture = tk.Button(self.category_frame, text="화면 속 이미지 저장",
                                      command=self.show_image_capture_screen)
        btn_image_capture.pack(pady=10, fill="x")

        btn_manage_functions = tk.Button(self.category_frame, text="저장된 함수 관리", # 텍스트 변경
                                         command=self.show_manage_functions_screen)
        btn_manage_functions.pack(pady=10, fill="x")

        btn_set_order = tk.Button(self.category_frame, text="순서 정하기",
                                  command=self.show_set_order_screen)
        btn_set_order.pack(pady=10, fill="x")

    def clear_content_frame(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_default_content(self):
        self.clear_content_frame()
        label = tk.Label(self.content_frame, text="기능을 선택해주세요.", bg="white")
        label.pack(expand=True, fill="both")

    # --- 학습 진행 상황 업데이트 함수 ---
    def update_training_status(self, message, progress_percent):
        # 이 함수는 다른 스레드에서 호출될 수 있으므로, Tkinter 업데이트는 mainloop에 예약해야 합니다.
        self.root.after(0, self._update_training_status_ui, message, progress_percent)

    def _update_training_status_ui(self, message, progress_percent):
        if self.training_status_label:
            self.training_status_label.config(text=message)
        
        if self.training_progress_bar:
            # 간단한 텍스트 프로그레스 바 구현
            bar_length = 20 # 바 길이 (문자 수)
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            self.training_progress_bar.config(text=f"[{bar}] {progress_percent}%")
        
        self.root.update_idletasks() # UI 강제 업데이트

    # --- 1. '화면 속 이미지 저장' 기능 ---
    def show_image_capture_screen(self):
        self.clear_content_frame()
        
        label_info = tk.Label(self.content_frame, text="화면 속 이미지를 캡처하여\nFaster R-CNN 학습 데이터로 만듭니다.", bg="white")
        label_info.pack(pady=10)

        btn_select_image = tk.Button(self.content_frame, text="이미지 선택하기",
                                     command=self.start_capture_thread)
        btn_select_image.pack(pady=10)

        self.captured_image_label = tk.Label(self.content_frame, bg="white")
        self.captured_image_label.pack(pady=5)
        
        self.image_size_label = tk.Label(self.content_frame, text="", bg="white")
        self.image_size_label.pack(pady=2)

        self.save_retry_frame = tk.Frame(self.content_frame, bg="white")
        self.save_retry_frame.pack(pady=10)

        self.btn_retry = tk.Button(self.save_retry_frame, text="다시하기", command=self.start_capture_thread)
        self.btn_save = tk.Button(self.save_retry_frame, text="저장하기", command=self.save_captured_image_for_training)
        
        self.save_retry_frame.pack_forget()

        # 학습 진행 상황 표시를 위한 라벨 추가
        self.training_status_label = tk.Label(self.content_frame, text="학습 대기 중...", bg="white", fg="blue")
        self.training_status_label.pack(pady=5)

        self.training_progress_bar = tk.Label(self.content_frame, text="", bg="white", font=("Courier", 10))
        self.training_progress_bar.pack(pady=2)


    def start_capture_thread(self):
        self.root.after(100, self.root.withdraw) # UI 숨기기를 메인 스레드에 예약
        time.sleep(0.1) 
        threading.Thread(target=self._initiate_capture_mode).start()

    def _initiate_capture_mode(self):
        self.is_capturing = True
        self.capture_window = tk.Toplevel(self.root)
        self.capture_window.attributes("-fullscreen", True) 
        self.capture_window.attributes("-alpha", 0.3)      
        self.capture_window.attributes("-topmost", True)   

        self.capture_canvas = tk.Canvas(self.capture_window, bg="black", highlightthickness=0)
        self.capture_canvas.pack(fill="both", expand=True)

        self.rect_id = None
        self.start_x = self.start_y = 0

        self.capture_canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.capture_canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.capture_canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        self.capture_window.focus_force()

    def _on_mouse_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.capture_canvas.delete(self.rect_id)
        self.rect_id = self.capture_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def _on_mouse_drag(self, event):
        self.capture_canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def _on_mouse_release(self, event):
        self.is_capturing = False
        self.capture_window.destroy() 
        self.root.after(100, self.root.deiconify) # UI 보이기를 메인 스레드에 예약

        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y

        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        if max_x - min_x > 0 and max_y - min_y > 0: 
            self.capture_area = (min_x, min_y, max_x, max_y)
            self._perform_screenshot()
        else:
            messagebox.showwarning("캡처 실패", "유효한 영역이 선택되지 않았습니다.")
            self.show_image_capture_screen()

    def _perform_screenshot(self):
        if self.capture_area:
            screenshot = ImageGrab.grab(bbox=self.capture_area)
            
            if not os.path.exists("temp_images"):
                os.makedirs("temp_images")
            self.current_captured_image_path = "temp_images/current_capture.png"
            screenshot.save(self.current_captured_image_path)

            display_image = screenshot.copy()
            max_width = int(self.content_frame.winfo_width() * 0.9)
            if display_image.width > max_width:
                display_image.thumbnail((max_width, max_width))
            
            self.captured_image_tk = ImageTk.PhotoImage(display_image)
            self.captured_image_label.config(image=self.captured_image_tk)
            self.image_size_label.config(text=f"크기: {screenshot.width}x{screenshot.height}")

            self.btn_retry.pack(side="left", padx=5)
            self.btn_save.pack(side="right", padx=5)
            self.save_retry_frame.pack(pady=10)
        else:
            messagebox.showerror("오류", "캡처할 영역이 지정되지 않았습니다.")
            self.show_image_capture_screen()

    def save_captured_image_for_training(self):
        if not self.current_captured_image_path:
            messagebox.showwarning("경고", "캡처된 이미지가 없습니다.")
            return

        func_name = simpledialog.askstring("함수 이름 입력", "이 이미지 탐색 함수의 이름을 입력하세요:")
        if not func_name:
            messagebox.showwarning("취소", "함수 저장이 취소되었습니다.")
            return

        if func_name in self.object_detector_manager.saved_functions:
            if not messagebox.askyesno("경고", f"'{func_name}' 함수가 이미 존재합니다. 덮어쓰시겠습니까?"):
                return
        
        # 학습은 별도의 스레드에서 시작
        threading.Thread(target=self.object_detector_manager.train_object_detection_model, 
                         args=(self.current_captured_image_path, func_name)).start()

        # 학습이 시작되었음을 UI에 표시
        self.update_training_status("학습 시작 중...", 0)
        self.save_retry_frame.pack_forget() # 저장/다시하기 버튼 숨김


    # --- 2. '저장된 함수 관리' 기능 (수정됨) ---
    def show_manage_functions_screen(self):
        self.clear_content_frame()
        
        label_title = tk.Label(self.content_frame, text="저장된 함수 관리", bg="white", font=("Arial", 12, "bold"))
        label_title.pack(pady=10)

        # 새 마우스 이동 함수 추가 버튼
        btn_add_mouse_move = tk.Button(self.content_frame, text="새 마우스 이동 함수 추가",
                                       command=self.save_mouse_move_function)
        btn_add_mouse_move.pack(pady=5)

        self.function_list_frame = tk.Frame(self.content_frame, bg="lightyellow", bd=1, relief="solid")
        self.function_list_frame.pack(pady=5, fill="both", expand=True, padx=10)

        self.populate_function_list()

    def save_mouse_move_function(self):
        """마우스 이동 함수를 좌표와 함께 저장합니다."""
        func_name = simpledialog.askstring("함수 이름 입력", "저장할 마우스 이동 함수의 이름을 입력하세요:")
        if not func_name:
            messagebox.showwarning("취소", "함수 저장이 취소되었습니다.")
            return

        if func_name in self.object_detector_manager.saved_functions:
            if not messagebox.askyesno("경고", f"'{func_name}' 함수가 이미 존재합니다. 덮어쓰시겠습니까?"):
                return

        x_str = simpledialog.askstring("좌표 입력", f"'{func_name}' 마우스 이동 함수의 X 좌표를 입력하세요:")
        y_str = simpledialog.askstring("좌표 입력", f"'{func_name}' 마우스 이동 함수의 Y 좌표를 입력하세요:")

        try:
            x, y = int(x_str), int(y_str)
            self.object_detector_manager.saved_functions[func_name] = {"type": "mouse_move", "x": x, "y": y}
            self.object_detector_manager.save_functions()
            messagebox.showinfo("저장 완료", f"'{func_name}' 마우스 이동 함수가 ({x}, {y})로 저장되었습니다.")
            self.populate_function_list() # 목록 새로고침
        except (ValueError, TypeError):
            messagebox.showerror("오류", "유효한 숫자를 입력하세요.")
            
    def populate_function_list(self):
        for widget in self.function_list_frame.winfo_children():
            widget.destroy()

        if not self.object_detector_manager.saved_functions:
            label_empty = tk.Label(self.function_list_frame, text="저장된 함수가 없습니다.", bg="lightyellow")
            label_empty.pack(pady=20)
            return

        for func_name, details in self.object_detector_manager.saved_functions.items():
            func_row_frame = tk.Frame(self.function_list_frame, bg="lightyellow")
            func_row_frame.pack(fill="x", pady=2, padx=5)

            func_text = func_name
            if details["type"] == "mouse_move":
                func_text += f" (X:{details['x']}, Y:{details['y']})"
            elif details["type"] == "image_search":
                func_text += " (이미지 탐색)"

            func_label = tk.Label(func_row_frame, text=func_text, bg="lightyellow", anchor="w")
            func_label.pack(side="left", expand=True, fill="x")

            btn_delete = tk.Button(func_row_frame, text="삭제", command=lambda name=func_name: self.delete_function(name))
            btn_delete.pack(side="right")

    def delete_function(self, func_name):
        if messagebox.askyesno("삭제 확인", f"'{func_name}' 함수를 정말 삭제하시겠습니까?"):
            if func_name in self.object_detector_manager.saved_functions:
                details = self.object_detector_manager.saved_functions[func_name]
                if details["type"] == "image_search":
                    model_path = details.get("model_path")
                    if model_path and os.path.exists(model_path):
                        try:
                            os.remove(model_path)
                            print(f"Deleted model file: {model_path}")
                        except Exception as e:
                            print(f"Error deleting model file {model_path}: {e}")

                del self.object_detector_manager.saved_functions[func_name]
                self.object_detector_manager.save_functions()
                messagebox.showinfo("삭제 완료", f"'{func_name}' 함수가 삭제되었습니다.")
                self.populate_function_list()
            else:
                messagebox.showerror("오류", f"'{func_name}' 함수를 찾을 수 없습니다.")


    # --- 3. '순서 정하기' 기능 (수정됨) ---
    def show_set_order_screen(self):
        self.clear_content_frame()

        self.content_frame.grid_columnconfigure(0, weight=2)
        self.content_frame.grid_columnconfigure(1, weight=6)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self.function_selection_frame = tk.Frame(self.content_frame, bg="lightblue", bd=2, relief="groove")
        self.function_selection_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.order_definition_frame = tk.Frame(self.content_frame, bg="lightgreen", bd=2, relief="groove")
        self.order_definition_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        # 순서 목록을 담을 Canvas와 Scrollbar
        self.ordered_functions_canvas = tk.Canvas(self.order_definition_frame, bg="lightgreen")
        self.ordered_functions_canvas.pack(side="top", fill="both", expand=True) # Top으로 변경
        self.ordered_functions_scrollbar = tk.Scrollbar(self.order_definition_frame, orient="vertical", command=self.ordered_functions_canvas.yview)
        self.ordered_functions_scrollbar.pack(side="right", fill="y")
        self.ordered_functions_canvas.configure(yscrollcommand=self.ordered_functions_scrollbar.set)
        # Configure scrollregion on canvas content update
        self.ordered_functions_canvas.bind('<Configure>', lambda e: self.ordered_functions_canvas.configure(scrollregion = self.ordered_functions_canvas.bbox("all")))
        
        self.ordered_functions_inner_frame = tk.Frame(self.ordered_functions_canvas, bg="lightgreen")
        self.ordered_functions_canvas.create_window((0,0), window=self.ordered_functions_inner_frame, anchor="nw")

        # Configure columns for the inner frame to control layout
        # 0번 열: 순서 번호 (고정 너비)
        self.ordered_functions_inner_frame.grid_columnconfigure(0, weight=0, minsize=30) 
        # 1번 열: 함수 이름 (유동적인 너비, 남은 공간을 채움)
        self.ordered_functions_inner_frame.grid_columnconfigure(1, weight=1) 
        # 2번 열: 삭제 버튼 (고정 너비)
        self.ordered_functions_inner_frame.grid_columnconfigure(2, weight=0, minsize=30) 

        self.populate_available_functions()
        self.populate_ordered_functions()

        # "순서 시작하기" 버튼 추가
        # 이 버튼은 canvas 아래에 위치해야 함
        btn_start_sequence = tk.Button(self.order_definition_frame, text="순서 시작하기",
                                        command=self.start_sequence, bg="lightcoral", fg="white", font=("Arial", 10, "bold"))
        btn_start_sequence.pack(pady=10, fill="x", padx=5) # pack()으로 변경


    def populate_available_functions(self):
        for widget in self.function_selection_frame.winfo_children():
            widget.destroy()
        
        label_title = tk.Label(self.function_selection_frame, text="함수 목록", bg="lightblue", font=("Arial", 10, "bold"))
        label_title.pack(pady=5)

        default_functions = [
            "마우스 드래그 (X축 10)",
            "클릭하기",
            "더블 클릭하기",
            "프로그램 종료하기"
        ]
        
        # 저장된 모든 함수 (이미지 탐색 및 마우스 이동)를 가져옵니다.
        saved_function_names = list(self.object_detector_manager.saved_functions.keys())

        all_functions = default_functions + saved_function_names
        
        self.selected_function = tk.StringVar(self.function_selection_frame)
        if all_functions:
            self.selected_function.set(all_functions[0])
            func_option_menu = tk.OptionMenu(self.function_selection_frame, self.selected_function, *all_functions)
            func_option_menu.pack(pady=5, fill="x", padx=5)
        else:
            tk.Label(self.function_selection_frame, text="사용 가능한 함수 없음", bg="lightblue").pack(pady=5)

        btn_add_to_order = tk.Button(self.function_selection_frame, text="순서 추가하기", command=self.add_function_to_order)
        btn_add_to_order.pack(pady=5, fill="x", padx=5)

        btn_clear_order = tk.Button(self.function_selection_frame, text="모두 지우기", command=self.clear_all_ordered_functions)
        btn_clear_order.pack(pady=5, fill="x", padx=5)

    def add_function_to_order(self):
        func_name = self.selected_function.get()
        if func_name:
            self.order_functions.append(func_name)
            self.populate_ordered_functions()
        else:
            messagebox.showwarning("경고", "추가할 함수를 선택해주세요.")

    def populate_ordered_functions(self):
        for widget in self.ordered_functions_inner_frame.winfo_children():
            widget.destroy()
        
        if not self.order_functions:
            tk.Label(self.ordered_functions_inner_frame, text="순서가 비어있습니다.\n함수를 추가해주세요.", bg="lightgreen").grid(row=0, column=0, columnspan=3, pady=20)
            return

        for i, func_name in enumerate(self.order_functions):
            # 순서 번호 레이블
            label_order = tk.Label(self.ordered_functions_inner_frame, text=f"{i+1}.", bg="lightgreen")
            label_order.grid(row=i, column=0, sticky="w", padx=(5, 0), pady=2) 

            display_name = func_name
            # 저장된 함수인 경우 상세 정보 추가
            if func_name in self.object_detector_manager.saved_functions:
                details = self.object_detector_manager.saved_functions[func_name]
                if details["type"] == "mouse_move":
                    display_name += f" (X:{details['x']}, Y:{details['y']})"
                elif details["type"] == "image_search":
                    display_name += " (이미지 탐색)"

            # 함수 이름 레이블: 고정된 너비 (약 30 문자 정도의 공간)
            # 이 'width' 값은 실제 UI의 크기와 텍스트 길이에 따라 조절해야 합니다.
            # 대략적인 픽셀 너비를 맞추려면 'width' 대신 `winfo_width()`를 사용하거나
            # 폰트 크기를 고려한 픽셀 단위 계산이 필요할 수 있습니다.
            label_func = tk.Label(self.ordered_functions_inner_frame, text=display_name, bg="lightgreen", anchor="w", width=30) 
            label_func.grid(row=i, column=1, sticky="ew", padx=5, pady=2) 

            # 삭제 버튼
            btn_delete_single = tk.Button(self.ordered_functions_inner_frame, text="X", fg="red", command=lambda idx=i: self.delete_single_ordered_function(idx))
            btn_delete_single.grid(row=i, column=2, sticky="e", padx=(0, 5), pady=2) 
        
        # 캔버스 스크롤 영역 업데이트
        self.ordered_functions_inner_frame.update_idletasks()
        self.ordered_functions_canvas.config(scrollregion=self.ordered_functions_canvas.bbox("all"))

    def delete_single_ordered_function(self, index):
        if messagebox.askyesno("삭제 확인", f"'{self.order_functions[index]}' 함수를 순서에서 삭제하시겠습니까?"):
            del self.order_functions[index]
            self.populate_ordered_functions()

    def clear_all_ordered_functions(self):
        if messagebox.askyesno("모두 지우기 확인", "정말 모든 순서를 지우시겠습니까?"):
            self.order_functions = []
            self.populate_ordered_functions()

    # --- 4. 기본 동작 함수 (실제 실행 로직) ---
    def execute_function(self, func_name):
        """
        정의된 함수 이름을 받아 실제 동작을 수행합니다.
        """
        # 저장된 함수인지 먼저 확인
        if func_name in self.object_detector_manager.saved_functions:
            details = self.object_detector_manager.saved_functions[func_name]
            if details["type"] == "mouse_move":
                x = details['x']
                y = details['y']
                self.object_detector_manager._move_mouse_absolute(x, y)
                return
            elif details["type"] == "image_search":
                self.object_detector_manager.find_image_and_move_mouse(func_name)
                return
        
        # 기본 내장 함수 처리
        if func_name == "마우스 드래그 (X축 10)":
            current_x, current_y = pyautogui.position()
            pyautogui.mouseDown(current_x, current_y)
            pyautogui.moveTo(current_x + 10, current_y, duration=0.2)
            pyautogui.mouseUp(current_x + 10, current_y)

        elif func_name == "클릭하기":
            pyautogui.click()

        elif func_name == "더블 클릭하기":
            pyautogui.doubleClick()
        
        elif func_name == "프로그램 종료하기":
            self.root.after(0, self.root.quit) # 메인 스레드에서 quit 호출 예약
        
        else:
            # 순서 실행 중이 아닐 때만 팝업
            if not self.is_sequence_running:
                messagebox.showwarning("알 수 없음", f"알 수 없는 함수: {func_name}")

    def start_sequence(self):
        """
        정의된 순서대로 함수들을 실행합니다.
        """
        if not self.order_functions:
            messagebox.showwarning("경고", "실행할 순서가 없습니다. 함수를 추가해주세요.")
            return

        if messagebox.askyesno("순서 실행", "정의된 순서를 실행하시겠습니까?"):
            self.is_sequence_running = True # 순서 실행 시작 플래그 설정
            # UI가 멈추지 않도록 새 스레드에서 실행
            threading.Thread(target=self._run_sequence_in_thread).start()

    def _run_sequence_in_thread(self):
        # UI 숨기기 (메인 스레드에서 실행되도록 after 메서드 사용)
        self.root.after(0, self.root.withdraw) 
        messagebox.showinfo("순서 실행", "자동화 순서가 시작됩니다. 작업 완료 전까지는 마우스를 움직이지 마세요.")

        try:
            for i, func_name in enumerate(self.order_functions):
                print(f"Executing step {i+1}: '{func_name}'") 
                
                # '프로그램 종료하기' 함수는 스레드 내에서 직접 호출하면 안 됩니다.
                # 메인 스레드에서 Tkinter 이벤트 루프를 종료하도록 처리해야 합니다.
                if func_name == "프로그램 종료하기":
                    self.root.after(0, self.root.quit) # 메인 스레드에서 quit 호출 예약
                    break # 종료 함수가 호출되면 더 이상 순서 진행 안 함
                
                self.execute_function(func_name)
                time.sleep(1) # 각 동작 후 1초 대기 (조절 가능)
            
            # 모든 순서가 정상적으로 완료되었을 때만 완료 메시지 출력
            # '프로그램 종료하기'로 인해 순서가 중단된 경우 메시지를 출력하지 않음
            if self.root.winfo_exists(): # Tkinter 창이 파괴되지 않았다면
                messagebox.showinfo("순서 완료", "모든 순서가 성공적으로 실행되었습니다.")

        except Exception as e:
            messagebox.showerror("순서 실행 오류", f"순서 실행 중 예상치 못한 오류 발생: {e}\n순서 실행을 중단합니다.")
        finally:
            self.is_sequence_running = False # 순서 실행 종료 플래그 해제
            # UI 다시 표시 (메인 스레드에서 실행되도록 after 메서드 사용)
            if self.root.winfo_exists(): # Tkinter 창이 아직 존재한다면 (프로그램 종료가 호출되지 않았다면)
                self.root.after(0, self.root.deiconify)


if __name__ == "__main__":
    # 필요한 디렉토리 생성
    os.makedirs("temp_images", exist_ok=True)
    os.makedirs("faster_rcnn_dataset/images", exist_ok=True)
    os.makedirs("faster_rcnn_dataset/labels", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    root = tk.Tk()
    app = AutomationApp(root)
    root.mainloop()