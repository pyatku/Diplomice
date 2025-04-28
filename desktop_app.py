import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from keras.models import load_model  # type: ignore
import os

# Укажите свой путь
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:\\Users\\МаРу\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\PyQt5\\Qt5\\plugins'

class IceDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Создание карт толщины льда")

        # Установка размеров окна
        self.window_width = 1000
        self.window_height = 600
        self.center_window()

        # Настройка сетки
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=1)
        master.rowconfigure(1, weight=1)
        master.rowconfigure(2, weight=0)

        # Цвета классов (RGB)
        self.colors = {
            0: (128,128,128), # серый
            1: (0, 255, 0),   # зеленый
            2: (0, 0, 255),   # синий 
            3: (0,255,255),   # голубой
            4: (255, 0, 255), # розовый
            5: (255, 255, 0), # желтый 
        }

        # Подписи классов
        self.labels = {
            0: "Нилас",
            1: "Молодой лед",
            2: "Однолетний лед",
            3: "Многолетний лед",
            4: "Вода",
            5: "Суша"
        }

        # Загрузка модели (укажите свой путь)
        self.model = load_model('C:\\Users\\МаРу\\Desktop\\дети\\диплом\\мы\\unet_model_v1_lr0.001_bs16.h5')

        # Создание кнопок
        self.load_btn = tk.Button(master, text="Загрузить изображение",
                                  command=self.load_image, height=2, width=20,
                                  bg="#4CAF50", fg="white", font=("Arial", 12))
        self.process_btn = tk.Button(master, text="Создать маску",
                                     command=self.process_image, height=2, width=20,
                                     bg="#4CAF50", fg="white", font=("Arial", 12))

        # Области для изображений и масок
        self.input_img_label = tk.Label(master)
        self.real_mask_label = tk.Label(master)
        self.pred_mask_label = tk.Label(master)
        self.legend_label = tk.Label(master)

        # Размещение элементов
        self.load_btn.grid(row=0, column=0, padx=10, pady=10)
        self.process_btn.grid(row=0, column=2, padx=10, pady=10)

        self.input_img_label.grid(row=1, column=0, padx=10, pady=10)
        self.real_mask_label.grid(row=1, column=1, padx=10, pady=10)
        self.pred_mask_label.grid(row=1, column=2, padx=10, pady=10)

        self.legend_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="W")

        # Пути к папкам (укажите свои)
        self.images_dir = r'C:\Users\МаРу\Desktop\дети\диплом\наработки апрель\images\датасет\valid'
        self.masks_dir = r'C:\Users\МаРу\Desktop\дети\диплом\наработки апрель\images\датасет\valid\masks'

        # Для хранения путей
        self.current_image_path = None
        print(self.model.summary())

    def center_window(self):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2
        self.master.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение",
                                               initialdir=self.images_dir,
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.original_image = Image.open(file_path)
            if self.original_image.mode == 'RGBA':
                self.original_image = self.original_image.convert('RGB')
            img = self.original_image.resize((256, 256))
            tk_img = ImageTk.PhotoImage(img)
            self.input_img_label.configure(image=tk_img)
            self.input_img_label.image = tk_img

            # Загрузить соответствующую маску
            self.load_real_mask(file_path)

    def load_real_mask(self, image_path):

        # Получаем имя файла без расширения
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Путь к папке с масками - папка "masks" внутри папки с изображениями
        images_folder = os.path.dirname(image_path)
        masks_folder = os.path.join(images_folder, 'masks')

        # Путь к маске с расширением .png
        mask_path = os.path.join(masks_folder, filename + '_mask.png')

        if not os.path.exists(mask_path):
            print(f"Маска не найдена для изображения: {filename}")
            self.real_mask_label.configure(image='')  # Очистить
            self.real_mask_label.image = None
            return

        # Открываем маску
        mask_img = Image.open(mask_path)

        # Если маска не RGB, конвертируем
        if mask_img.mode != 'RGB':
            mask_img = mask_img.convert('RGB')

        # Отобразим маску
        mask_display = mask_img.resize((256, 256))
        tk_mask = ImageTk.PhotoImage(mask_display)
        self.real_mask_label.configure(image=tk_mask)
        self.real_mask_label.image = tk_mask

    def apply_color_map(self, mask_array):
        h, w = mask_array.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self.colors.items():
            colored_mask[mask_array == class_id] = color

        return colored_mask

    def process_image(self):
        if hasattr(self, 'original_image'):
            # Препроцессинг
            img_array = np.array(self.original_image.resize((128, 128))) / 255.0
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)

            # Предсказание
            prediction = self.model.predict(img_array)[0]
            print("shape -", prediction.shape)
            print("min, max -", prediction.min(), prediction.max())

            # Постобработка
            mask = np.argmax(prediction, axis=-1)

            # Применяем цветовую карту
            colored_mask = self.apply_color_map(mask)

            # Создаем PIL Image из маски
            mask_img = Image.fromarray(colored_mask)
            
            # Изменяем размер маски до 256x256
            mask_img = mask_img.resize((256, 256), Image.NEAREST)  # Изменяем размер здесь
            
            # Создаем легенду и отображаем ее
            legend_img = self.create_legend()
            tk_legend = ImageTk.PhotoImage(legend_img)
            self.legend_label.configure(image=tk_legend)
            self.legend_label.image = tk_legend

            # Отображаем предсказанную маску
            tk_mask = ImageTk.PhotoImage(mask_img)
            self.pred_mask_label.configure(image=tk_mask)
            self.pred_mask_label.image = tk_mask

    def create_legend(self):
        # Создаем легенду с цветами и подписями в один ряд
        font_size = 12
        font = ImageFont.load_default()
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            pass

        # Измеряем ширину каждой подписи
        label_widths = [font.getlength(label) for label in self.labels.values()]

        # Рассчитываем общую ширину легенды
        total_width = sum(label_widths) + len(self.labels) * 50  # Приблизительная общая ширина

        legend_height = 40
        legend_img = Image.new('RGB', (int(total_width), legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend_img)

        # Параметры для отрисовки
        rect_size = 15
        padding = 5
        x = padding
        y = (legend_height - rect_size) // 2

        for class_id in sorted(self.colors.keys()):
            color = self.colors[class_id]
            label = self.labels[class_id]

            # Рисуем цветной квадрат
            draw.rectangle([int(x), int(y), int(x) + rect_size, int(y) + rect_size], fill=color)

            # Рисуем тире
            dash_x = x + rect_size + padding
            draw.text((dash_x, y), " - ", fill=(0, 0, 0), font=font)

            # Рисуем подпись справа от тире
            text_x = dash_x + font.getlength(" - ")
            draw.text((text_x, y), label, fill=(0, 0, 0), font=font)

            # Обновляем позицию x для следующего элемента (горизонтальное размещение)
            bbox = draw.textbbox((text_x, y), label, font=font)
            x = text_x + (bbox[2] - bbox[0]) + padding * 3

        return legend_img

    def add_legend(self, image):
        return image

if __name__ == '__main__':
    root = tk.Tk()
    app = IceDetectorApp(root)
    root.mainloop()
