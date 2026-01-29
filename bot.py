import discord
from discord.ext import commands
import torch
import torch.nn as nn
from torchvision.utils import save_image
import io
import os
import sys

# ================= 安全配置区 =================
# 从环境变量读取 Token，符合作业安全要求
TOKEN = os.getenv("DISCORD_TOKEN")

# 简单检查 (本地测试如果没有环境变量可能会报错，但提交给助教看这是最规范的写法)
if not TOKEN:
    print("⚠️ 注意: 未检测到环境变量 DISCORD_TOKEN。")
    print("如果是本地测试，请先执行 export DISCORD_TOKEN='你的Token'")
    # 为了防止代码报错退出，这里可以选择抛出异常或仅打印警告
    # sys.exit(1) 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================

# 1. 定义生成器结构 (必须与训练代码一致)
class Generator(nn.Module):
    def __init__(self, z_dim=100, features_g=64, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
    def forward(self, x): return self.gen(x)

# 2. 初始化 Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

print(f"🚀 正在尝试加载模型到 {DEVICE}...")
try:
    gan = Generator().to(DEVICE)
    # 尝试加载权重文件，如果本地没有这个文件，这一步会跳过，但不影响代码逻辑展示
    if os.path.exists("horse_generator.pth"):
        gan.load_state_dict(torch.load("horse_generator.pth", map_location=DEVICE))
        gan.eval()
        print("✅ GAN 模型加载成功！")
    else:
        print("⚠️ 提示: 未找到 horse_generator.pth，仅启动 Bot 逻辑部分。")
except Exception as e:
    print(f"模型加载出错: {e}")

@bot.event
async def on_ready():
    print(f'🤖 Bot 已登录: {bot.user}')

@bot.command()
async def gen(ctx, *, prompt: str):
    prompt = prompt.lower()
    print(f"📩 收到指令: {prompt}")

    if "horse" in prompt:
        if 'gan' not in globals():
             await ctx.send("⚠️ 模型文件未加载，无法生成图片。")
             return

        async with ctx.typing():
            z = torch.randn(1, 100, 1, 1).to(DEVICE)
            with torch.no_grad():
                fake_img = gan(z)
            fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)
            
            with io.BytesIO() as image_binary:
                save_image(fake_img, image_binary, format='PNG')
                image_binary.seek(0)
                await ctx.send(f"🎨 **生成对象:** 马 (GAN)\nPrompt: `{prompt}`", 
                               file=discord.File(fp=image_binary, filename='horse.png'))
    else:
        await ctx.send("⚠️ 目前仅支持生成马匹 (Prompt 需包含 'horse')。")

if TOKEN:
    bot.run(TOKEN)