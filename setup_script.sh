#!/bin/bash
set -e

echo "📦 Starte vollständigen HunyuanVideo Setup..."

# ----------------------------------------
# Step 0: Cache leeren
echo "🧹 Leere Cache-Ordner (conda, pip, HF)..."
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/pip
rm -rf ~/.conda/pkgs
rm -rf ~/.cache/torch_extensions
rm -rf /root/.cache/huggingface

# ----------------------------------------
# Step 1: Miniconda installieren oder korrigieren
if [ ! -d "$HOME/miniconda3" ] || [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "⬇️  Miniconda wird (neu) installiert..."
  rm -rf $HOME/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda3
  source ~/miniconda3/etc/profile.d/conda.sh
  conda init
  source ~/.bashrc
else
  echo "✅ Miniconda vorhanden"
  source ~/miniconda3/etc/profile.d/conda.sh
fi

# ----------------------------------------
# Step 2: Conda-Umgebung erstellen
if conda info --envs | grep -q "^hunyuan"; then
  echo "⚠️  Conda-Umgebung 'hunyuan' existiert bereits"
else
  echo "🐍 Erstelle Conda-Umgebung 'hunyuan'..."
  conda create -n hunyuan python=3.10 -y
fi

echo "📂 Aktiviere Umgebung..."
conda activate hunyuan

# ----------------------------------------
# Step 3: Git LFS installieren
if ! command -v git-lfs &> /dev/null; then
  echo "🔧 Installiere Git LFS..."
  apt update && apt install git-lfs -y
  git lfs install
else
  echo "✅ Git LFS ist installiert"
fi

# ----------------------------------------
# Step 4: PyTorch & Libraries installieren
echo "⚙️  Installiere PyTorch 2.6.0 + CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "📚 Installiere NLP/Utility Pakete..."
pip install transformers accelerate einops scipy loguru safetensors imageio diffusers

# ----------------------------------------
# Step 5: HunyuanVideo Repo holen
if [ ! -d "HunyuanVideo" ]; then
  echo "📁 Klone HunyuanVideo Repo..."
  git clone https://github.com/Tencent/HunyuanVideo.git
fi
cd HunyuanVideo

mkdir -p ckpts

# ----------------------------------------
# Step 6: Hauptmodell via git lfs holen
if [ ! -d "ckpts/hunyuan-video-t2v-720p" ]; then
  echo "📥 Lade Hauptmodell via Git LFS..."
  git lfs clone https://huggingface.co/Tencent/HunyuanVideo ckpts/
fi

# ----------------------------------------
# Step 7: Textencoder - CLIP
if [ ! -d "ckpts/text_encoder_2" ]; then
  echo "🎯 Lade CLIP Textencoder..."
  huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2 --repo-type model
fi

# ----------------------------------------
# Step 8: Textencoder - LLaVA / LLM
if [ ! -d "ckpts/text_encoder" ]; then
  echo "🧠 Lade LLaVA LLM Textencoder..."
  huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava

  echo "🔧 Preprocessiere LLaVA Encoder..."
  python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir ./ckpts/llava \
    --output_dir ./ckpts/text_encoder
fi

# ----------------------------------------
# Step 9: FlashAttention installieren
echo "⚡ Installiere FlashAttention..."
pip install ninja
pip install flash-attn --no-build-isolation --no-cache-dir

# ----------------------------------------
# Step 10: XFormers + xfuser
pip install xformers xfuser==0.4.0

# ----------------------------------------
# Step 11: Modell-Dateien prüfen
echo ""
if [ ! -f "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" ]; then
  echo "❌ WICHTIG: Modelldatei fehlt! Prüfe deinen Download!"
  exit 1
fi

# ----------------------------------------
echo ""
echo "✅ Setup abgeschlossen!"
echo "👉 Conda-Umgebung 'hunyuan' ist aktiv."
echo "🎬 Beispiel:"
echo "python sample_video.py --prompt 'A cat walking' --video-size 360 640 --video-length 8 --infer-steps 10 --save-path ./results/test.mp4 --use-cpu-offload"
