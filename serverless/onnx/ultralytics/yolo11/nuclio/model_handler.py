import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolov11-nms-640.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
            print(f"[INFO] Model loaded: {model}, Providers: {providers}")
        except Exception as e:
            raise Exception(f"[ERROR] Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32)
            im /= 255

            inp = {self.input_details[0]: im}
            detections = self.model.run(self.output_details, inp)[0]

            if len(detections) == 0:
                print("[WARN] No detections returned from model.")
                return None

            print(">>> Raw detections shape:", detections.shape)
            print(">>> Raw detections sample:", detections[0][:6])

            detections = detections[0]
            boxes = detections[:, 0:4]
            scores = detections[:, 4]
            labels = detections[:, 5].astype(int)

            print(f">>> Parsed boxes: {boxes.shape}, labels: {labels.shape}, scores: {scores.shape}")
            pad = np.array([dwdh[0], dwdh[1], dwdh[0], dwdh[1]])
            print(">>> pad:", pad, "ratio:", ratio)

            boxes = boxes - pad
            boxes = boxes / ratio

            h, w = inputs.shape[:2]
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

            boxes[:, 0], boxes[:, 2] = np.minimum(boxes[:, 0], boxes[:, 2]), np.maximum(boxes[:, 0], boxes[:, 2])
            boxes[:, 1], boxes[:, 3] = np.minimum(boxes[:, 1], boxes[:, 3]), np.maximum(boxes[:, 1], boxes[:, 3])

            boxes = boxes.round().astype(np.int32)

            print(">>> Final boxes (after resize correction):", boxes[:3])
            return [boxes, labels, scores]

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None

    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes = detections[0]
            labels = detections[1]
            scores = detections[2]

            print(f">>> Threshold filtering: {threshold}")
            print(f">>> Total detections: {len(boxes)}")

            for label, score, box in zip(labels, scores, boxes):
                xtl = max(int(box[0]), 0)
                ytl = max(int(box[1]), 0)
                xbr = min(int(box[2]), w)
                ybr = min(int(box[3]), h)

                if xbr - xtl <= 1 or ybr - ytl <= 1:
                    print(f"[SKIP] Invalid box size: {(xtl, ytl, xbr, ybr)}")
                    continue

                if score >= threshold:
                    label_name = self.labels.get(label)
                    if label_name is None:
                        print(f"[SKIP] Unknown label id: {label}")
                        continue

                    result = {
                        "confidence": str(score),
                        "label": label_name,
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                    }

                    results.append(result)
                    
                    print("[RESULT]", result)

            if not results:
                print("[INFO] No detections passed the threshold.")
        else:
            print("[INFO] No detections to process.")

        return results

