import torch
import sys

print(f" [+] Рабочее окружение: Python {sys.version.split()[0]}")
print(f" [+] Сборка PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f" \e[1;32m[+] Успех! CUDA видит карту:\e[0m {torch.cuda.get_device_name(0)}")
    print(" [!] Система полностью готова к генерации через Qwen3.")
else:
    print(" \e[1;31m[!] Ошибка: Драйверы CUDA не задействованы.\e[0m")
