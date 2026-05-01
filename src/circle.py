import cv2
import numpy as np

def freehand_crop(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("画像読み込み失敗")
        return None

    clone = img.copy()

    drawing = False
    points = []
    result = None
    finished = False

    def draw_freehand(event, x, y, flags, param):
        nonlocal drawing, points, result, finished

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                points.append((x, y))
                temp = clone.copy()
                cv2.polylines(temp, [np.array(points)], False, (0, 255, 0), 2)
                cv2.imshow("image", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            points.append((x, y))

            pts = np.array(points, dtype=np.int32)

            mask = np.zeros(clone.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            result = cv2.bitwise_and(clone, clone, mask=mask)

            cv2.imshow("result", result)

            # ★ここで終了フラグ
            finished = True

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_freehand)

    while True:
        cv2.imshow("image", img)

        key = cv2.waitKey(1)

        # ESCで終了
        if key == 27:
            break

        # ウィンドウ閉じられたら終了
        if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            break

        # ★囲い終わったら即終了
        if finished:
            break

    cv2.destroyAllWindows()

    return result