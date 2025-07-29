import numpy as np
import os
import re
from tqdm import tqdm

BOARD_SIZE = 15
POS = 'abcdefghijklmno'

def move_to_coords(move_str):
    """将 SGF 坐标 (如 'hh') 转换为 (row, col) 元组 (如 (7, 7))"""
    try:
        col = POS.find(move_str[0])
        row = POS.find(move_str[1])
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
    except:
        return None
    return None

def final_parser(sgf_path):
    """
    最终版的解析器，结合了用户提供的代码思路。
    使用 ISO-8859-1 编码并用正则表达式直接提取棋步和结果。
    """
    try:
        with open(sgf_path, 'r', encoding='ISO-8859-1') as f:
            content = f.read()
    except Exception:
        return [], [], []

    # 提取对局结果
    result_match = re.search(r'RE\[([^\]]+)\]', content)
    if result_match:
        result_str = result_match.group(1).upper()
        if 'B+' in result_str:
            winner = 1  # Black wins
        elif 'W+' in result_str:
            winner = 2  # White wins
        else:
            winner = 0  # Draw
    else:
        winner = 0 # 默认为和棋

    # 正则表达式匹配 ;B[..] 或 ;W[..]
    moves = re.findall(r';(B|W)\[([a-o]{2})\]', content)
    
    if not moves:
        return [], [], []

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    states = []
    policies = []
    values = []

    for i, (color_str, move_str) in enumerate(moves):
        coords = move_to_coords(move_str)
        if coords is None:
            continue
        
        row, col = coords
        
        if board[row, col] != 0:
            continue

        current_state = np.copy(board)
        
        player = 1 if color_str == 'B' else 2
        opponent = 2 if player == 1 else 1
        
        model_input = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
        model_input[:, :, 0] = (current_state == player)
        model_input[:, :, 1] = (current_state == opponent)
        states.append(model_input)
        
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[row * BOARD_SIZE + col] = 1.0
        policies.append(policy)
        
        # 分配价值
        if winner == 0:
            values.append(0.0)
        elif winner == player:
            values.append(1.0)
        else:
            values.append(-1.0)

        board[row, col] = player

    return states, policies, values


def process_all_sgf_files(sgf_dir, output_file):
    """
    处理指定目录下所有的 SGF 文件，并保存为 .npz 文件。
    """
    all_states = []
    all_policies = []
    all_values = []
    
    sgf_files = [f for f in os.listdir(sgf_dir) if f.endswith('.sgf')]
    
    if not sgf_files:
        print(f"No .sgf files found in {sgf_dir}")
        return

    print(f"Found {len(sgf_files)} SGF files. Processing...")

    for filename in tqdm(sgf_files):
        sgf_path = os.path.join(sgf_dir, filename)
        states, policies, values = final_parser(sgf_path)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)

    if not all_states:
        print("No valid training data could be extracted.")
        return

    print(f"Processed {len(all_states)} total moves.")
    print(f"Saving data to {output_file}...")

    np.savez_compressed(output_file, 
                        states=np.array(all_states, dtype=np.float32), 
                        policies=np.array(all_policies, dtype=np.float32),
                        values=np.array(all_values, dtype=np.float32))
    
    print("Done.")

if __name__ == '__main__':
    SGF_DIRECTORY = 'sgf'
    OUTPUT_NPZ_FILE = 'gomoku_data.npz'
    process_all_sgf_files(SGF_DIRECTORY, OUTPUT_NPZ_FILE)
