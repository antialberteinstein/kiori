import os
import sys
import threading
import pygame

# Đảm bảo import kiori cục bộ thay vì bản pip
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cấu hình môi trường Threading / Forking an toàn cho macOS và PyTorch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["USE_TF"] = "0"

import torch
from transformers import pipeline

# Import chuẩn từ kiori
from kiori.agent import KioriAgent
from kiori.models import Action, ActionExample
from kiori.memory import MilvusLTM, ReplayBuffer

# Khởi tạo mô hình (1B - không cần prefix-fill, dùng transformers chat template)
print("Đang tải mô hình unsloth/gemma-3-1b-it...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Sử dụng thiết bị: {device}")

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

generator = pipeline(
    "text-generation",
    model="unsloth/gemma-3-1b-it",
    device=device,
    torch_dtype=dtype
)
print("Tải mô hình thành công!")

# 1. Kiến trúc Agent & RobotSim
class RobotSim:
    def __init__(self, grid_size=10, max_x=9, max_y=9):
        self.grid_size = grid_size
        self.x = grid_size//2
        self.y = grid_size//2
        self.max_x = max_x
        self.max_y = max_y

    def move_up(self, steps: int):
        '''Go up by number of steps'''
        self.y = max(0, self.y - steps)
        return f"Tiến lên {steps} bước. Tọa độ: ({self.x}, {self.y})"

    def move_down(self, steps: int):
        '''Go down by number of steps'''
        self.y = min(self.max_y, self.y + steps)
        return f"Lùi xuống {steps} bước. Tọa độ: ({self.x}, {self.y})"

    def move_left(self, steps: int):
        '''Turn left by number of steps'''
        self.x = max(0, self.x - steps)
        return f"Sang trái {steps} bước. Tọa độ: ({self.x}, {self.y})"

    def move_right(self, steps: int):
        '''Turn right by number of steps'''
        self.x = min(self.max_x, self.x + steps)
        return f"Sang phải {steps} bước. Tọa độ: ({self.x}, {self.y})"

# Khởi tạo Memory modules
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
db_path = os.path.join(log_dir, "pygame_robot_wo.db")

ltm = MilvusLTM(db_path=db_path, collection_name="robot_actions")
replay_buffer = ReplayBuffer.NO_REPLAY_BUFFER

# Khởi tạo đối tượng
robot = RobotSim(grid_size=10, max_x=9, max_y=9)
# chat_format=None → Kiori trả prompt thuần, callback dùng transformers pipeline tự áp template
agent = KioriAgent(ltm=ltm, replay_buffer=replay_buffer, threshold=0.65, max_copies=3)

# Đăng ký các hàm vào agent
agent.add_action(Action("move_up", "Move up", robot.move_up))
agent.add_action(Action("move_down", "Move down", robot.move_down))
agent.add_action(Action("move_left", "Move left", robot.move_left))
agent.add_action(Action("move_right", "Move right", robot.move_right))

# Thêm Few-Shot Examples vào Long-Term Memory (LTM)
examples = [
    ActionExample("move up", action_name="move_up", kwargs={"steps": 1}),
    ActionExample("move up 3 steps", action_name="move_up", kwargs={"steps": 3}),
    ActionExample("go up 2 steps", action_name="move_up", kwargs={"steps": 2}),

    ActionExample("move down", action_name="move_down", kwargs={"steps": 1}),
    ActionExample("go down by 2", action_name="move_down", kwargs={"steps": 2}),
    ActionExample("move down by 5", action_name="move_down", kwargs={"steps": 5}),
    
    ActionExample("move left", action_name="move_left", kwargs={"steps": 1}),
    ActionExample("turn left 1 step", action_name="move_left", kwargs={"steps": 1}),
    ActionExample("move left 3 steps", action_name="move_left", kwargs={"steps": 3}),
    
    ActionExample("move right", action_name="move_right", kwargs={"steps": 1}),
    ActionExample("move right 5", action_name="move_right", kwargs={"steps": 5}),
    ActionExample("turn right 2 steps", action_name="move_right", kwargs={"steps": 2}),
]
ltm.clear()
ltm.add_examples(examples)

# Biến trạng thái UI
log_message = "Kiori Agent ready! Enter command..."
is_thinking = False

# 2. Hàm chạy nền xử lý LLM (Background Thread)
def process_command_thread(user_text):
    global log_message, is_thinking
    
    def llm_callback(prompt: str) -> str:
        """
        Callback: nhận prompt thuần từ Kiori (không có chat template),
        wrap vào messages và dùng transformers pipeline để áp chat template tự động.
        Thêm 'Action:' vào cuối để hướng model sinh action thay vì tiếp tục pattern.
        """
        messages = [{"role": "user", "content": prompt}]
        
        outputs = generator(messages, max_new_tokens=128, temperature=0.2)
        llm_text = outputs[0]['generated_text'][-1]['content'].strip()
            
        print(f"\n[LLM Raw Output]: {llm_text}")
        return llm_text

    try:
        result = agent.run(user_text, llm_callback=llm_callback, max_retries=3)
        log_message = f"SUCCESS: {result}"
    except Exception as e:
        log_message = f"Kiori Error: {e}"
    finally:
        is_thinking = False

# 4. Giao diện Pygame
def main():
    global log_message, is_thinking
    pygame.init()
    
    # Cấu hình cửa sổ
    WIDTH, HEIGHT = 600, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Kiori Robot Simulator (Gemma 3 1B - No Chat Template)")
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    log_font = pygame.font.SysFont("Arial", 16)
    
    # Cấu hình Text Input
    input_box = pygame.Rect(50, 580, 500, 40)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_active
    text = ''
    
    running = True
    while running:
        # 60 FPS
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if not is_thinking:  # Khóa nhập khi đang xử lý
                    if event.key == pygame.K_RETURN:
                        if text.strip() != "":
                            # Bắt đầu xử lý lệnh ở Background Thread
                            is_thinking = True
                            log_message = "Model is thinking..."
                            threading.Thread(target=process_command_thread, args=(text,)).start()
                            text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
                        
        # Render
        screen.fill((30, 30, 30)) # Màu nền tối
        
        # --- Vẽ Grid 2D ---
        grid_width = 400
        grid_height = 400
        start_x = (WIDTH - grid_width) // 2
        start_y = 50
        cell_size = grid_width // robot.grid_size
        
        # Vẽ nền lưới
        pygame.draw.rect(screen, (50, 50, 50), (start_x, start_y, grid_width, grid_height))
        
        # Vẽ các ô
        for i in range(robot.grid_size):
            for j in range(robot.grid_size):
                rect = (start_x + i * cell_size, start_y + j * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)
                
        # Vẽ Robot
        robot_rect = (start_x + robot.x * cell_size, start_y + robot.y * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 200, 100), robot_rect)
        
        # --- Vẽ Text Input ---
        pygame.draw.rect(screen, color, input_box, 2)
        txt_surface = font.render(text, True, (255, 255, 255))
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        input_box.w = max(500, txt_surface.get_width() + 10)
        
        # Nhấp nháy con trỏ
        if time.time() % 1 > 0.5 and not is_thinking:
            cursor_pos = input_box.x + 5 + txt_surface.get_width()
            pygame.draw.line(screen, (255, 255, 255), (cursor_pos, input_box.y + 5), (cursor_pos, input_box.y + 35), 2)
            
        # --- Vẽ Log ---
        log_surface = log_font.render(log_message, True, (200, 200, 0) if is_thinking else (150, 255, 150))
        screen.blit(log_surface, (50, 640))
        
        # Chỉ dẫn
        help_text = font.render("Press Enter to send command", True, (150, 150, 150))
        screen.blit(help_text, (50, 550))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    import time
    main()
