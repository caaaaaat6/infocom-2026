import pulp
import os
import shutil

def get_solver():
    """智能地查找并返回一个可用的 CBC 求解器。"""
    # 优先级 1: 检查 HOMEBREW 的标准路径 (针对 macOS)
    homebrew_path = "/opt/homebrew/bin/cbc"
    if os.path.exists(homebrew_path):
        return pulp.COIN_CMD(path=homebrew_path, msg=False)
        
    # 优先级 2: 检查用户是否通过环境变量 CBC_PATH 指定了路径
    cbc_path_env = os.environ.get("CBC_PATH")
    if cbc_path_env and os.path.exists(cbc_path_env):
        return pulp.COIN_CMD(path=cbc_path_env, msg=False)

    # 优先级 3: 尝试 PuLP 默认的 PULP_CBC_CMD
    solver = pulp.PULP_CBC_CMD(msg=False)
    if solver.available():
        return solver

    # 优先级 4: 检查系统 PATH 中是否有 'cbc'
    if shutil.which("cbc"):
        return pulp.COIN_CMD(msg=False)

    # 如果都失败了，打印帮助信息
    # ...
    return None

# --- 在模块加载时，只初始化一次求解器 ---
CBC_SOLVER = get_solver()

print(CBC_SOLVER)








