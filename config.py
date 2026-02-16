# config.py

class Config:
    # ========================
    # 训练阶段控制
    # ========================
    STAGE = "pretrain"      # "pretrain" 或 "finetune"

    # ========================
    # 是否使用 Hessian
    # ========================
    USE_HESSIAN = False     # 预训练默认 False

    # ========================
    # Hessian 参数
    # ========================
    SIGMA = [0.5, 1, 1.5, 2]
    SPACING = [1, 1]
    TAU = 2