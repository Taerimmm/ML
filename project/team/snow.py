import pynput
from pynput.keyboard import Key, Controller
import time

mouse_drag = pynput.mouse.Controller()
mouse_button = pynput.mouse.Button

def mouse_move():
    default = mouse_drag.position
    print(default)

    # mouse_drag.position = (100,700)

    mouse_drag.press(mouse_button.left)
    mouse_drag.release(mouse_button.left)


class TestKeyboard: 
    def __init__(self): 
        self.keyboard = Controller() 

    def inputKey(self, key): 
        self.keyboard.press(key) 
        self.keyboard.release(key) 
    
    def inputKeyWithShift(self, key): 
        with self.keyboard.pressed(Key.shift): 
            self.keyboard.press(key) 
            self.keyboard.release(key) 
    
    def inputKeyWithControl(self, key): 
        with self.keyboard.pressed(Key.ctrl): 
            self.keyboard.press(key) 
            self.keyboard.release(key) 
    
    def inputKeyWith(self, with_key, key): 
        with self.keyboard.pressed(with_key): 
            self.keyboard.press(key) 
            self.keyboard.release(key) 
            
    def typeString(self, string): 
        self.keyboard.type(string) 
        
if __name__ == '__main__': 
    mouse_move()

    # kb = TestKeyboard() 
    
    # time.sleep(10) #테스트를 위해 5초 이후 입력이 시작되게 하였다 
    # kb.inputKey('a') 
    # kb.typeString('gozz`s tistory')
    # kb.inputKey(Key.enter) 
    # kb.inputKeyWithShift('b')
    
# (988, 980)
# (1171, 775)
# (688, 968)
# (1188, 1015)