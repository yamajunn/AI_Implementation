from pynput import keyboard
from pynput.keyboard import Controller
import time

# モールス符号の辞書
MORSE = {
    "a": [0, 1], "b": [1, 0, 0, 0], "c": [1, 0, 1, 0], "d": [1, 0, 0],
    "e": [0], "f": [0, 0, 1, 0], "g": [1, 1, 0], "h": [0, 0, 0, 0],
    "i": [0, 0], "j": [0, 1, 1, 1], "k": [1, 0, 1], "l": [0, 1, 0, 0],
    "m": [1, 1], "n": [1, 0], "o": [1, 1, 1], "p": [0, 1, 1, 0],
    "q": [1, 1, 0, 1], "r": [0, 1, 0], "s": [0, 0, 0], "t": [1],
    "u": [0, 0, 1], "v": [0, 0, 0, 1], "w": [0, 1, 1], "x": [1, 0, 0, 1],
    "y": [1, 0, 1, 1], "z": [1, 1, 0, 0], "1": [0, 1, 1, 1, 1],
    "2": [0, 0, 1, 1, 1], "3": [0, 0, 0, 1, 1], "4": [0, 0, 0, 0, 1],
    "5": [0, 0, 0, 0, 0], "6": [1, 0, 0, 0, 0], "7": [1, 1, 0, 0, 0],
    "8": [1, 1, 1, 0, 0], "9": [1, 1, 1, 1, 0], "0": [1, 1, 1, 1, 1]
}

key_controller = Controller()  # キーボード操作用
signals = []                   # 入力されたモールス信号
last_release_time = time.time()  # 最後にキーが離された時刻

def on_press(key):
    """
    キーが押されたときの処理
    """
    if key == keyboard.Key.shift:
        global press_start_time
        press_start_time = time.time()

def on_release(key):
    """
    キーが離されたときの処理
    """
    if key == keyboard.Key.shift:
        global signals, last_release_time
        last_release_time = time.time()
        duration = last_release_time - press_start_time
        signals.append(0 if duration < 0.15 else 1)
    elif key == keyboard.Key.esc:
        print("プログラムを終了します。")
        return False  # プログラム終了

def process_signals():
    """
    入力信号を処理して対応する文字を入力
    """
    global signals
    if time.time() - last_release_time > 0.4 and signals:
        for char, code in MORSE.items():
            if signals == code:
                key_controller.press(char)
                key_controller.release(char)
                break
        signals.clear()

# リスナーの設定と開始
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while listener.running:
        process_signals()  # 定期的に信号を処理
        time.sleep(0.1)
except KeyboardInterrupt:
    print("プログラムを終了します。")

listener.stop()
