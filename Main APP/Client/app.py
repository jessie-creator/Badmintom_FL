from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Canvas, Color, RoundedRectangle, Rectangle
import os
import csv
import webbrowser
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import random
import time
import uuid
import json
import requests

class FilePath():
    current_directory, phone_path = os.getcwd(), ''
    #current_directory, phone_path = '', '/storage/emulated/0/Android/data/ru.iiec.pydroid3/local_train_app_a53'

    ui_resource = {
            'font': os.path.join(current_directory, phone_path, 'jf-openhuninn-1.1/jf-openhuninn-1.1.ttf'),
            'xmas': os.path.join(current_directory, phone_path, 'ui_reference/Xmas.jpeg'),
            'logo': os.path.join(current_directory, phone_path, 'ui_reference/logo_c.png'),
            'home': os.path.join(current_directory, phone_path, 'ui_reference/home.png'),
            'easteregg': os.path.join(current_directory, phone_path, 'ui_reference/fairy.PNG'),
            'high_ball': os.path.join(current_directory, phone_path, 'ui_reference/high_ball.png'),
            'short_ball': os.path.join(current_directory, phone_path, 'ui_reference/short_ball.png'),
            'tutorial_high': os.path.join(current_directory, phone_path, 'ui_reference/tutorial_high.jpg'),
            'tutorial_short': os.path.join(current_directory, phone_path, 'ui_reference/tutorial_short.jpg')
            }

    data_path = {
            'train_dataset': os.path.join(current_directory, phone_path, 'train_dataset/'),
            'predict_dataset': os.path.join(current_directory, phone_path, 'predict_dataset/'),
            'model_high': os.path.join(current_directory, phone_path, 'local_model/model_high.pth'),
            'model_short': os.path.join(current_directory, phone_path, 'local_model/model_short.pth'),
            'downloaded_model_high': os.path.join(current_directory, phone_path, 'local_model/downloaded_model_high.pth'),
            'downloaded_model_short': os.path.join(current_directory, phone_path, 'local_model/downloaded_model_short.pth'),
            'uuid': os.path.join(current_directory, phone_path, 'uuid/uuid.txt')
            }

class SharedData():
    current_directory, phone_path = os.getcwd(), ''
    #current_directory, phone_path = '', '/storage/emulated/0/Android/data/ru.iiec.pydroid3/local_train_app_a53'

    shared_data = {
            'pitch_type': '',
            'data': [],
            'labels': [], 
            'mistake_types': [],
            'score': [],
            'last_score': [3, 2, 3, 4, 3],
            'recommend_score': [],
            'status': '',
            'predict_flag': 0,
            'train_acc': '',
            'train_acc_high': '',
            'train_acc_short': '',
            'debug': '',
            'update_fed': 0,
            'uuid': None
            }
    tutorial_video_list = {
            '0': 'https://youtu.be/9k8TpZne038',
            '1': 'https://reurl.cc/L4jkgX',
            '2': 'https://www.youtube.com/watch?v=AbHBCg83tOo',
            '3': 'https://www.youtube.com/watch?v=DZuKRbWrJkY&t=211s',
            '4': 'https://www.youtube.com/watch?v=sB-TVcnpmxU',
            '5': 'https://youtu.be/3KyLDK0mRuY?si=ZUKna8kAtKtqgLQ0',
            '6': 'https://www.youtube.com/watch?v=8mXaccKOYp4',
            '7': 'https://youtu.be/y3tSQkfTTDU?si=C8RfcN9oGKwNCheb'
            }

class Setting():
    ngrok_url = "http://9852-27-240-225-90.ngrok-free.app"
    ngrok_url = "http://127.0.0.1:8001"
    url_create_uuid = ngrok_url + "/create_uuid"
    url_upload_model = ngrok_url + "/upload_model"
    url_download_model = ngrok_url + "/download_model"
    url_run_backend_process = ngrok_url + "/run_backend_process"

    train_parameter = {
            #常數放置區
            'learning_rate': 0.01,
            'batch_size': 20,
            'numEpoch': 50,
            'output_dim': 5,
            'num_layers': 1,
            'dropout_rate': 0.1,
            'input_size': 6,
            'train_ratio': 0.8,
            'test_ratio': 0.2
            }


class DataPreprocessing():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_files = []
        self.n = 6
        self.timeLenth = 60

    def slope(self, input_data):
        slope_col = np.zeros(len(input_data),dtype=float)
        for i in range(len(input_data) - 1):
            for j in range(1,4):
                temp = input_data[i+1][j] - input_data[i][j]
                slope_col[i] += temp**2
            slope_col[i] = slope_col[i]**0.5
        return slope_col

    def ma(self, slope_col, n):
        slope_ma = np.zeros(len(slope_col),dtype=float)
        for i in range(n,len(slope_col)-n):
            for j in range(-n,n+1):
                slope_ma[i] += slope_col[i+j]
            slope_ma[i] /= float(2*n+1)
        return slope_ma

    def data_cut(self, input_data, save_data, hit_type, n):
        slope_col = self.slope(input_data)
        slope_avg = np.average(slope_col)
        slope_ma = self.ma(slope_col, n)
        for i in range(60,len(slope_col)-50):
            if ( slope_ma[i] > slope_avg ) and ( slope_col[i]==max(slope_col[i-50:i+50]) ) : # 找到可能峰值
                start = 0  # 向前&向後找起點
                end = 0
                while i+start > (50+n) :
                    start -= 1
                    if slope_ma[i+start] <= slope_avg:
                        break
                while i+end < (len(slope_col)-50-n) :
                    end += 1
                    if slope_ma[i+end] <= slope_avg:
                        break
                if hit_type == 'high' and end-start > 40 and slope_ma[i] > 15: # 依球種分類存入 save_data
                    save_data.append(input_data[i-45:i+15, [1,2,3,5,6,7]])
                elif hit_type == 'short' and end-start > 20 and slope_ma[i] > 15: # 挑球
                    save_data.append(input_data[i-35:i+15, [1,2,3,5,6,7]])

    # Removes null bytes from the input file and returns a sanitized version of the file.
    def sanitize_file(self, input_file_path):
        sanitized_content = ""
        with open(input_file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            sanitized_content = content.replace('\x00', '')
        return sanitized_content

    def cut_data(self, data_path, pitch_type, n, timeLenth):
        try:
            all_files = os.listdir(data_path)
            # print('all_files in cut data(): ', all_files)
            #txt_files = [file_name for file_name in all_files if file_name.endswith('.txt') and hittype_name in file_name and len(file_name)>=30]

            SharedData.shared_data['labels'] = np.empty((0, 5), dtype=int)
            SharedData.shared_data['data'] = np.empty((0, timeLenth, 6), dtype=float)

            for t, input_filename in enumerate(all_files):
                input_file_path = os.path.join(data_path, input_filename)

                # Sanitize the file by removing null bytes
                sanitized_content = self.sanitize_file(input_file_path)
                lines = sanitized_content.split('\n')

                # Starting from the last line, move upwards until a complete line (with 7 commas) is found
                while lines and lines[-1].count(",") != 7:
                    lines = lines[:-1]

                # Load the (potentially modified) data into a numpy array
                input_data = np.loadtxt(lines, delimiter=",", dtype=float)
                save_data = []
                self.data_cut(input_data, save_data, pitch_type, n)
                save_data = np.array(save_data).astype(float)

                SharedData.shared_data['data'] = np.concatenate((SharedData.shared_data['data'], save_data), axis=0)

                label_values = np.array([int(ch) for ch in input_filename[6:11]])  # 轉換為整數陣列
                label_values = label_values[np.newaxis, :]  # 增加一個維度以使其成為二維陣列

                for i in range(len(save_data)):
                    SharedData.shared_data['labels'] = np.vstack((SharedData.shared_data['labels'], label_values))
        except Exception as e:
            print(f'Data cut failed: {e}')

class CustomFunc():
    def custom_output(x):
        return x

    def custom_loss(y_true, y_pred):
        loss = torch.mean((y_true - y_pred) ** 2)
        return loss

    def custom_accuracy(y_true, y_pred):
        return torch.mean((torch.abs(y_true - y_pred) < 0.5).float(), dtype=torch.float32)

    def custom_prediction(y_pred):
        pred = torch.mean(y_pred, axis=0)
        return np.round(pred) - 1

# Local_Model: GRU
class CustomGRUModel(nn.Module):
    def __init__(self, input_size, units, output_dim, num_layers, dropout_rate):
        super(CustomGRUModel, self).__init__()
        self.gru_layers = nn.ModuleList(
                [nn.GRU(input_size if i == 0 else units, units, batch_first=True) for i in range(num_layers)]
                )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(units, output_dim)

    def forward(self, x):
        for gru_layer in self.gru_layers:
            x, _ = gru_layer(x)
            x = self.dropout(x) # Apply dropout after each GRU layer
        # Use the output of the last GRU layer (x) for prediction
        x = self.dense(x[:, -1, :]) # Selecting the output from the last time step for prediction
        return CustomFunc.custom_output(x)

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Early Stopping 早停法
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stopping utility.

        :param patience: 數量的 epochs，在這些 epochs 中如果損失沒有改善，則提前停止訓練。
        :param min_delta: 被認為是改善的最小變化量。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ReTrain():

    def retrain(self, pitch_type, n, timeLength, numEpoch=10):
        try:
            #start_time = time.time()
            obj = DataPreprocessing()
            obj.cut_data(FilePath.data_path['train_dataset'], pitch_type, n, timeLength)
            #end_time = time.time()
            #pre_data_time = start_time - end_time
            #print('data_pre_time = ', pre_data_time)
        except Exception as e:
            print(f'DataPreprocessing failed {e}')
        try:
            #start_time = time.time()
            x_train, x_test, y_train, y_test = train_test_split(SharedData.shared_data['data'], SharedData.shared_data['labels'], test_size=Setting.train_parameter['test_ratio'], random_state=24)

            x_train = x_train.reshape(x_train.shape)
            x_test = x_test.reshape(x_test.shape)
            train_dataset = MyDataset(x_train, y_train)
            test_dataset = MyDataset(x_test, y_test)

            train_dataloader = DataLoader(train_dataset, batch_size=Setting.train_parameter['batch_size'], shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=Setting.train_parameter['batch_size'], shuffle=False)

            # Instantiate the custom model
            model = CustomGRUModel(input_size=Setting.train_parameter['input_size'], units=64, output_dim=Setting.train_parameter['output_dim'], num_layers=Setting.train_parameter['num_layers'], dropout_rate=Setting.train_parameter['dropout_rate'])
            model_path = FilePath.data_path[f'model_{pitch_type}']  # 模型參數檔案的路徑
            model.load_state_dict(torch.load(model_path))
            optimizer = torch.optim.Adam(model.parameters(), lr=Setting.train_parameter['learning_rate'])
            loss_fn = CustomFunc.custom_loss

            # Train with epoch
            early_stopping = EarlyStopping(patience=10, min_delta=0.01)
            for epoch in range(numEpoch):
                model.train()
                for x_batch, y_batch in train_dataloader:
                    optimizer.zero_grad()
                    y_pred = model(x_batch)
                    loss = loss_fn(y_batch, y_pred)
                    loss.backward()
                    optimizer.step()

                # 評估
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    total_accuracy = 0
                    for x_batch, y_batch in test_dataloader:
                        y_pred = model(x_batch)
                        total_loss += loss_fn(y_batch, y_pred).item()
                        total_accuracy += CustomFunc.custom_accuracy(y_batch, y_pred).item()
                avg_loss = total_loss / len(test_dataloader)
                avg_accuracy = total_accuracy / len(test_dataloader)
                print(f'Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {avg_accuracy}')
                # Early Stopping 檢查
                early_stopping(avg_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            SharedData.shared_data[f'train_acc_{pitch_type}'] = str(round(avg_accuracy, 4))
            # 定義模型的路徑和檔名
            model_path = FilePath.data_path[f'model_{pitch_type}']
            #end_time = time.time()
            #train_time = start_time - end_time
            #print('train_time = ', train_time)
            # 保存模型參數到檔案
            print(model_path)
            torch.save(model.state_dict(), model_path)
            print('save successful')
        except Exception as e:
            print(f'local train failed: {e}')
            SharedData.shared_data['status'] = f'local train failed: {e}'
        return

class Prediction():
    def gru_pred(self, file_path, file_name):
        SharedData.shared_data['score'] = np.empty((0, 5), dtype=int)
        if file_name[0] == 'h':
            pitch_type, n, timeLength = 'high', 6, 60
        else:
            pitch_type, n, timeLength = 'short', 5, 50

        try:
            obj = DataPreprocessing()
            #print('create datapre success\n')
            #print('file_path: ', file_path)
            obj.cut_data(file_path, pitch_type, n, timeLength)
            #print('cut_data success\n')
        except Exception as e:
            pass
        #print(f'failed {e}')
        try:
            x_train, x_test, y_train, y_test = train_test_split(SharedData.shared_data['data'], SharedData.shared_data['labels'], test_size=1, random_state=24)
            x_test = x_test.reshape(x_test.shape)
            model = CustomGRUModel(input_size=Setting.train_parameter['input_size'], units=64, output_dim=Setting.train_parameter['output_dim'], num_layers=Setting.train_parameter['num_layers'], dropout_rate=Setting.train_parameter['dropout_rate'])
            model_path = FilePath.data_path[f'model_{pitch_type}']  # 模型參數檔案的路徑
            model.load_state_dict(torch.load(model_path))
            test_dataset = MyDataset(x_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in test_dataloader:
                    y_pred = model(x_batch)
                    y_pred = CustomFunc.custom_prediction(y_pred)
            SharedData.shared_data['score'] = y_pred
            #print(y_pred)
            #print('score: ', SharedData.shared_data['score'])
        except Exception as e:
            pass
        #print(f'failed {e}')
        return

# 定义一个BoxLayout，它将在KV字符串中使用
class MyBoxLayout(BoxLayout):
    pass

class MarkPopup_EasterEgg(Popup):
    pic_easteregg = FilePath.ui_resource['easteregg']
class MarkPopup(Popup):
    pass
#def (self):
    #    app = App.get_running_app()
    #    app.root.current = 'update'

class CustomMarkPopup(Popup):
    def __init__(self, content_text='', **kwargs):
        super().__init__(**kwargs)

        # 在彈出視窗中添加標籤(Label)元件，顯示指定的內容
        self.content_label = Label(text=content_text)
        self.content_label.font_size = '20sp'  # 設置字體大小
        self.content_label.bold = True  # 設置為粗體
        self.content_label.color = (0, 0, 1, 1)  # 設置文字顏色為藍色
        self.content_label.size_hint_y = None
        self.content_label.height = self.content_label.texture_size[1]

        self.content = self.content_label

class MarkPopup_Update(Popup):
    def set_update_flag(self):
        SharedData.shared_data['update_fed'] = 1

class MyButton(Button):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
        self.bold = True
        self.color = (68/255, 124/255, 104/255, 1)
        self.background_color = (155/255, 197/255, 183/255, 0)
        with self.canvas.before:
            Color(155/255, 197/255, 183/255, 1)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)
        self.font_name = 'custom_font'

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class MyButton_c(Button):
    def __init__(self, **kwargs):
        super(MyButton_c, self).__init__(**kwargs)
        self.bold = True
        self.color1 = (68/255, 124/255, 104/255, 1)  # 状态1的颜色
        self.color2 = (1, 1, 1, 1)  # 状态2的颜色
        self.background_color1 = (155/255, 197/255, 183/255, 1)  # 状态1的背景颜色
        self.background_color2 = (68/255, 124/255, 104/255, 1)  # 状态2的背景颜色
        with self.canvas.before:
            Color(*self.background_color1)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size)
        self.background_color = (155/255, 197/255, 183/255, 0)
        self.bind(pos=self.update_rect, size=self.update_rect)
        self.font_name = 'custom_font'
        self.state_costom = 'state1'  # 记录按钮状态
        self.update_color(self.color1, self.background_color1)

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_release(self):
        if self.state_costom == 'state1':
            self.set_state2()
        elif self.state_costom == 'state2':
            self.set_state1()

    def set_state1(self):
        self.state_costom = 'state1'
        self.update_color(self.color1, self.background_color1)

    def set_state2(self):
        self.state_costom = 'state2'
        self.update_color(self.color2, self.background_color2)

    def update_color(self, color, background_color):
        self.color = color

# Declare both screens
class SplashScreen(Screen):
    pic_xmas = FilePath.ui_resource['xmas']
    def on_enter(self):
        Clock.schedule_once(self.switch_to_main_screen, 8)

    def switch_to_main_screen(self, dt):
        self.manager.current = 'home'

class HomeScreen(Screen):
    pic_logo = FilePath.ui_resource['logo']

class SwingScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    pic_logo = FilePath.ui_resource['logo']
    pic_high_ball = FilePath.ui_resource['high_ball']
    pic_short_ball = FilePath.ui_resource['short_ball']

# 系統評分
class EvaluateScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    def toggle_file_chooser(self):
        # 获取可折叠的 BoxLayout
        file_chooser_layout = self.ids.file_chooser_layout
        screen_height = Window.size[1]

        # 设置展开的高度为屏幕高度的0.25
        if self.ids.file_chooser_layout.height == 0:
            self.ids.file_chooser_layout.height = screen_height * 0.2
        else:
            self.ids.file_chooser_layout.height = 0  # 收起

    def expand_buttons(self, instance):
        menu_layout = self.ids.menu_layout
        if menu_layout.height == 0:
            menu_layout.height = screen_height * 0.1
        else:
            menu_layout.height = 0

    def prediction(self):
        if self.ids.file_chooser.selection:
            selected_file = self.ids.file_chooser.selection[0]
            try:
                file_path, file_name = os.path.split(selected_file)
                file_path = os.path.join(FilePath.current_directory, FilePath.phone_path, file_path)
                obj = Prediction()
                obj.gru_pred(file_path, file_name)
            except Exception as e:
                pass
            #print(f'gru prediction failed{e}')

    def change_label(self):
        scores = SharedData.shared_data['score']
        if scores == []:
            popup = MarkPopup(title='Hint')
            popup.ids.popup_label.text = 'Please choose the file first.'
            popup.open()

        else:
            scores = scores.tolist()
            #print('my_scores', scores)
            self.ids.score_1.text = str(int(scores[0])+1) + '分'
            self.ids.score_2.text = str(int(scores[1])+1) + '分'
            self.ids.score_3.text = str(int(scores[2])+1) + '分'
            self.ids.score_4.text = str(int(scores[3])+1) + '分'
            self.ids.score_5.text = str(int(scores[4])+1) + '分'

# 教練打分
class MarkScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    mark_sheet = [[0] * 5 for _ in range(5)]
    score_sheet = []

    def on_pre_enter(self, *args):
        # Initiate mark_sheet before enter MarkScreen
        self.mark_sheet = [[0] * 5 for _ in range(5)]
        for child in self.walk():
            if isinstance(child, MyButton_c):
                child.set_state1()

    def rename_file(self):
        if self.ids.file_chooser.selection:
            selected_file = self.ids.file_chooser.selection[0]
            try:
                file_path, file_name = os.path.split(selected_file)
                score_as_str = ''.join(str(idx+1) for idx in self.score_sheet)
                new_file_name = file_name[:5] + "_" + score_as_str + file_name[5:]
                new_file_path = os.path.join(file_path, new_file_name)
                os.rename(selected_file, new_file_path)
                #self.show_message("File renamed successfully!")
            except Exception as e:
                pass    
            #self.show_message(f"Error: {str(e)}")
        #else:
            #self.show_message("Please select a file.")

    def check_conditions(self, flag):
        # Check conditions and show the appropriate popup
        for col in range(len(self.mark_sheet[0])):
            count_ones = sum(row[col] for row in self.mark_sheet)
            if count_ones != 1:
                # 如果某一列没有且仅有一个1，则不满足条件
                self.show_popup()
                return
        # 满足条件，切换到 'swing' 屏幕
        self.score_sheet = np.argmax(self.mark_sheet, axis=1)
        self.rename_file()
        app = App.get_running_app()
        if flag == 1:
            app.root.current = 'swing'
        else:
            app.root.current = 'mark'

    def show_popup(self):
        MarkPopup(title= 'Hint').open()

    def on_button_click(self, col, row):
        # 按钮被点击时，根据按钮的位置更新 mark_sheet
        if self.mark_sheet[row][col] == 1:
            self.mark_sheet[row][col] = 0
        else:
            self.mark_sheet[row][col] = 1

    def toggle_file_chooser(self):
        # 获取可折叠的 BoxLayout
        file_chooser_layout = self.ids.file_chooser_layout
        screen_height = Window.size[1]

        # 设置展开的高度为屏幕高度的0.25
        if self.ids.file_chooser_layout.height == 0:
            self.ids.file_chooser_layout.height = screen_height * 0.2
        else:
            self.ids.file_chooser_layout.height = 0  # 收起

# 最近一次的分數
class ScoreScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    def on_enter(self):
        scores = SharedData.shared_data['last_score']
        print('my_scores', scores)
        self.ids.score_1.text = str(int(scores[0])) + '分'
        self.ids.score_2.text = str(int(scores[1])) + '分'
        self.ids.score_3.text = str(int(scores[2])) + '分'
        self.ids.score_4.text = str(int(scores[3])) + '分'
        self.ids.score_5.text = str(int(scores[4])) + '分'

class UpdateScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    def on_pre_enter(self, *args):
        label = self.ids.status_label
        label.text = ''

    def model_retrain(self):
        #debug_label = self.ids.debug_label
        label = self.ids.status_label
        try:
            obj = ReTrain()
            label.text = 'wait for training'
        except Exception as e:
            pass
        try:
            print('create success')
            obj.retrain('high', 6, 60, 50)
            obj.retrain('short', 5, 50, 50)
            label.text = 'Accuracy for 高遠球 is ' + str(100*float(SharedData.shared_data['train_acc_high'])) + '%\n\nAccuracy for 挑球 is ' + str(100*float(SharedData.shared_data['train_acc_short'])) + '%'
            print('高遠球模型的準確率為：' + str(100*float(SharedData.shared_data['train_acc_high'])) + '%\n挑球模型的準確率為：' + str(100*float(SharedData.shared_data['train_acc_short'])) + '%')
        except Exception as e:
            print(f'local train failed{e}')
            label.text = f'local train failed{e}'
            #ebug_label.text = TestApp.shared_data['debug']

    def show_popup(self):
        SharedData.shared_data['update_fed'] = 0
        MarkPopup_Update(title= 'Hint').open()
        return

    def aggregate(self, pitch_type):
        model1_weights = torch.load(FilePath.data_path[f'model_{pitch_type}'])
        model2_weights = torch.load(FilePath.data_path[f'downloaded_model_{pitch_type}'])

        avg_weights = {
                key: (model1_weights[key] + model2_weights[key]) / 2
                for key in model1_weights.keys()
                }
        torch.save(avg_weights, FilePath.data_path[f'model_{pitch_type}'])

    def model_fedml(self):
        label = self.ids.status_label
        if SharedData.shared_data['update_fed'] == 0:
            self.show_popup()
            print('after show popup update')
        elif SharedData.shared_data['update_fed'] == 1:
            SharedData.shared_data['update_fed'] = 0
            with open(FilePath.data_path[f'uuid'], 'r') as file:
                uuid_str = file.readline().strip()  # 讀取並去除字串兩側的空白符
            print("從 uuid.txt 讀取的 UUID 為:", uuid_str)
            #print(str(uuid))
            #uuid_str = str(uuid)
            uuid_json_str = uuid_str.lstrip('b\'').rstrip('\\n\'')
            uuid_data = json.loads(uuid_json_str)
            uuid_value = uuid_data['uuid']
            print("提取的UUID值:", uuid_value)

            # 打開模型檔案，讀取二進位模式
            with open(FilePath.data_path[f'model_high'], "rb") as high_model_file, open(FilePath.data_path[f'model_short'], "rb") as short_model_file:
                files = {
                        "high_model_file": (FilePath.data_path[f'model_high'], high_model_file, "application/octet-stream"),
                        "short_model_file": (FilePath.data_path[f'model_short'], short_model_file, "application/octet-stream")
                        }
                response = requests.post(Setting.url_upload_model+"?uuid="+uuid_value, files=files)
                if response.status_code != 200:
                    popup = MarkPopup(title='Hint')
                    popup.ids.popup_label.text = f'{response.status_code}: Model upload failed.'
                    popup.open()
            response = requests.post(Setting.url_run_backend_process)
            if response.status_code != 200:
                popup = MarkPopup(title='Hint')
                popup.ids.popup_label.text = f'{response.status_code}: Model aggregation failed.'
                popup.open()

            response = requests.get(Setting.url_download_model+"?model_type=high")
            if response.status_code == 200:
                with open(FilePath.data_path[f'downloaded_model_high'], 'wb') as f:
                    f.write(response.content)
                self.aggregate('high')

            else:
                popup = MarkPopup(title='Hint')
                popup.ids.popup_label.text = f'{response.status_code}: Model for high ball download failed.'
                popup.open()
            response = requests.get(Setting.url_download_model+"?model_type=short")
            if response.status_code == 200:
                # 如果成功下載了檔案，你可以對應處理下載的檔案內容
                with open(FilePath.data_path[f'downloaded_model_short'], 'wb') as f:
                    f.write(response.content)
                self.aggregate('short')
            else:
                popup = MarkPopup(title='Hint')
                popup.ids.popup_label.text = f'{response.status_code}: Model for short ball download failed.'
                popup.open()

            obj = ReTrain()
            print('create success')
            obj.retrain('high', 6, 60)
            obj.retrain('short', 5, 50)
            label.text = 'Accuracy for 高遠球 is ' + str(100*float(SharedData.shared_data['train_acc_high'])) + '%\n\nAccuracy for 挑球 is ' + str(100*float(SharedData.shared_data['train_acc_short'])) + '%'
            print('高遠球模型的準確率為：' + str(100*float(SharedData.shared_data['train_acc_high'])) + '%\n挑球模型的準確率為：' + str(100*float(SharedData.shared_data['train_acc_short'])) + '%')


class TutorialScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    pic_tutorial_high = FilePath.ui_resource['tutorial_high']
    pic_tutorial_short = FilePath.ui_resource['tutorial_short']
    seed = int(time.time())
    random.seed(seed)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        SharedData.shared_data['recommend_score'] = SharedData.shared_data['last_score']

    def tutorial_high_link(instance):
        min_score = min(SharedData.shared_data['recommend_score'])
        min_index = np.argmin(SharedData.shared_data['recommend_score'])
        if random.random() > 0.8 or min_index == 0:
            link = SharedData.tutorial_video_list[str(random.randint(0, 4))]
        elif min_index == 1:
            link = SharedData.tutorial_video_list['0']
        elif min_index == 2 or min_index == 3:
            link = SharedData.tutorial_video_list['4']
        elif min_index == 4:
            if random.random() > 0.5:
                link = SharedData.tutorial_video_list['3']
            else:
                link = SharedData.tutorial_video_list['0']
        webbrowser.open(link)

    def tutorial_short_link(instance):
        min_score = min(SharedData.shared_data['recommend_score'])
        min_index = np.argmin(SharedData.shared_data['recommend_score'])
        if random.random() > 0.8 or min_index == 0 or min_index == 3:
            link = SharedData.tutorial_video_list[str(random.randint(5, 7))]
        elif min_index == 1:
            link = SharedData.tutorial_video_list['5']
        elif min_index == 2 or min_index == 4:
            if random.random() > 0.5:
                link = SharedData.tutorial_video_list['5']
            else:
                link = SharedData.tutorial_video_list['6']
        webbrowser.open(link)

#class MarkSheet(Screen):

class TutorialMarkScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    mark_sheet = [[0] * 5 for _ in range(5)]

    def on_pre_enter(self, *args):
        # When enter the screen, initiate mark_sheet
        self.mark_sheet = [[0] * 5 for _ in range(5)]
        for child in self.walk():
            if isinstance(child, MyButton_c):
                child.set_state1()

    def check_conditions(self):
        # Check if exist 1 and only exist 1 entry with value in every
        # Otherwise show the popup (hint)
        for col in range(5):
            count_ones = sum(row for row in self.mark_sheet[col])
            if count_ones != 1:
                MarkPopup(title= 'Hint').open()
                return
        #if easter_egg:
        #    self.show_popup(easter_egg)
        SharedData.shared_data['recommend_score'] = np.argmax(self.mark_sheet, axis=1)
        print(sum(SharedData.shared_data['recommend_score']))
        if sum(SharedData.shared_data['recommend_score']) == 20:
            MarkPopup_EasterEgg(title = 'Congratualation').open()
        app = App.get_running_app()
        app.root.current = 'tutorial'


    def on_button_click(self, col, row):
        # When button is clicked, update mark_sheet
        if self.mark_sheet[col][row] == 1:
            self.mark_sheet[col][row] = 0
        else:
            self.mark_sheet[col][row] = 1

class VideoListScreen(Screen):
    pic_home = FilePath.ui_resource['home']
    def open_link(instance, video_name):
        webbrowser.open(SharedData.tutorial_video_list[video_name])


# 设置全局字体
LabelBase.register(name='custom_font',
        fn_regular=FilePath.ui_resource['font'])

# 获取屏幕宽度和高度
screen_width = Window.width
screen_height = Window.height

my_layout = MyBoxLayout()
my_layout.screen_width = screen_width
my_layout.screen_height = screen_height

Builder.load_file('app.kv')
#Builder.load_file('/storage/emulated/0/Android/data/ru.iiec.pydroid3/local_train_app_a53/app.kv')

class TestApp(App):
    def build(self):
        # Create uuid
        response = requests.get(Setting.url_create_uuid)
        if response.status_code == 200:
            uuid = response.content
            with open(FilePath.data_path[f'uuid'], 'w') as file:
                file.write(str(uuid))  # 將 UUID 轉換為字串並寫入檔案

        # Create the screen manager
        sm = ScreenManager()
        sm.add_widget(SplashScreen(name='splash'))
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(SwingScreen(name='swing'))
        sm.add_widget(EvaluateScreen(name='evaluate'))
        sm.add_widget(MarkScreen(name='mark'))
        sm.add_widget(ScoreScreen(name='score'))
        sm.add_widget(TutorialScreen(name='tutorial'))
        sm.add_widget(UpdateScreen(name='update'))
        sm.add_widget(TutorialMarkScreen(name='tutorial_mark'))
        sm.add_widget(VideoListScreen(name='video_list'))
        return sm

if __name__ == '__main__':
    TestApp().run()
