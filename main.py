from tkinter import Tk, filedialog
from src.predict import predict
from src.circle import freehand_crop
from src.drawing_graph import extract_1d_signal_from_clean_mask, create_graph, df2csv
from pathlib import Path

def select_folder_path():
    root = Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="フォルダを選択"
    )

    if not folder_path:
        print("フォルダが選択されていません")
        return None

    print("選択されたフォルダ:", folder_path)
    return folder_path

def main():
    file_path = select_folder_path()
    predict_image_paths = predict(file_path, "trained_model.pth")
    for predict_image_path in predict_image_paths:
        result = freehand_crop(predict_image_path)
        digital_data, ymin, ymax, xmin, xmax = extract_1d_signal_from_clean_mask(result)
        name = Path(predict_image_path).stem
        csv_path = Path("outputs/csv_files") / f"digital_data_{name}.csv"
        df2csv(digital_data, csv_path)
        create_graph(csv_path, ymin, ymax, xmin, xmax)

if __name__ == "__main__":
    main()
